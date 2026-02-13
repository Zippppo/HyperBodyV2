from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import autocast
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from .hyperbolic_network import HyperBodyNet
from .lorentz_loss import LorentzTreeRankingLoss
from .organ_hierarchy import compute_tree_distance_matrix, load_organ_hierarchy


class nnUNetTrainerHyperBody(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Hyperbolic branch hyperparameters
        self.hyp_embed_dim = 48
        self.hyp_curv = 1.0
        self.hyp_margin = 0.4
        self.hyp_samples_per_class = 128
        self.hyp_num_negatives = 16
        self.hyp_weight = 0.25
        self.hyp_t_start = 2.0
        self.hyp_t_end = 0.1
        self.hyp_warmup_epochs = 60
        self.hyp_freeze_epochs = 50
        self.hyp_text_lr_ratio = 0.01
        self.hyp_text_grad_clip = 0.1
        self.hyp_distance_mode = "graph"
        self.hyp_direction_mode = "random"
        self.hyp_text_embedding_path = None
        self.hyp_min_radius = 0.1
        self.hyp_max_radius = 2.0

        self.hyp_lr = 1e-3

        self.hyp_loss = None
        self.optimizer_hyp_head = None
        self.optimizer_hyp_emb = None
        self.lr_scheduler_hyp_head = None
        self.lr_scheduler_hyp_emb = None

    def _ensure_logging_key(self, key: str) -> None:
        if key not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging[key] = []

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

        default_embed_dim = arch_init_kwargs.get("features_per_stage", [32])[0]
        network = HyperBodyNet(
            backbone=backbone,
            num_classes=num_output_channels,
            embed_dim=default_embed_dim,
            curv=1.0,
            class_depths=None,
            min_radius=0.1,
            max_radius=2.0,
            direction_mode="random",
            text_embedding_path=None,
        )
        if not enable_deep_supervision:
            network.hyperbolic_mode = False
        return network

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

        self._set_batch_size_and_oversample()
        self.num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )

        class_depths = self._load_class_depths_from_hierarchy()

        backbone = get_network_from_plans(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.label_manager.num_segmentation_heads,
            allow_init=True,
            deep_supervision=self.enable_deep_supervision,
        )

        embed_dim = self.configuration_manager.network_arch_init_kwargs["features_per_stage"][0]
        self.hyp_embed_dim = embed_dim

        self.network = HyperBodyNet(
            backbone=backbone,
            num_classes=self.label_manager.num_segmentation_heads,
            embed_dim=embed_dim,
            curv=self.hyp_curv,
            class_depths=class_depths,
            min_radius=self.hyp_min_radius,
            max_radius=self.hyp_max_radius,
            direction_mode=self.hyp_direction_mode,
            text_embedding_path=self.hyp_text_embedding_path,
        ).to(self.device)

        if self._do_i_compile():
            self.print_to_log_file("Using torch.compile...")
            self.network = torch.compile(self.network)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)

        self.loss = self._build_loss()
        self.hyp_loss = self._build_hyp_loss().to(self.device)

        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True

    def _load_class_depths_from_hierarchy(self):
        tree_path = self._resolve_dependency_path("tree.json")
        if tree_path is None:
            self.print_to_log_file("tree.json not found. Falling back to uniform label embedding radii.")
            return None

        class_names = self._class_names_by_index(self.dataset_json)
        return load_organ_hierarchy(str(tree_path), class_names)

    def _build_hyp_loss(self) -> LorentzTreeRankingLoss:
        class_names = self._class_names_by_index(self.dataset_json)
        num_classes = self.label_manager.num_segmentation_heads

        graph_path = self._resolve_dependency_path("graph_distance_matrix.pt")
        if graph_path is not None:
            tree_dist_matrix = torch.load(graph_path, map_location="cpu", weights_only=False).float()
        else:
            tree_path = self._resolve_dependency_path("tree.json")
            if tree_path is not None:
                tree_dist_matrix = compute_tree_distance_matrix(str(tree_path), class_names)
            else:
                # Last-resort fallback so trainer can still run functional tests.
                tree_dist_matrix = torch.ones(num_classes, num_classes, dtype=torch.float32)
                tree_dist_matrix.fill_diagonal_(0)

        if tuple(tree_dist_matrix.shape) != (num_classes, num_classes):
            raise ValueError(
                f"graph distance matrix shape mismatch: expected ({num_classes}, {num_classes}), "
                f"got {tuple(tree_dist_matrix.shape)}"
            )

        return LorentzTreeRankingLoss(
            tree_dist_matrix=tree_dist_matrix,
            margin=self.hyp_margin,
            curv=self.hyp_curv,
            num_samples_per_class=self.hyp_samples_per_class,
            num_negatives=self.hyp_num_negatives,
            t_start=self.hyp_t_start,
            t_end=self.hyp_t_end,
            warmup_epochs=self.hyp_warmup_epochs,
        )

    def _resolve_dependency_path(self, filename: str) -> Path | None:
        candidates: list[Path] = []

        if nnUNet_raw is not None:
            candidates.append(Path(nnUNet_raw) / self.plans_manager.dataset_name / filename)

        # Fallback for this repository layout: root/Dataset/*.json|*.pt
        # Walk up to find the repo root (.git marker) instead of hardcoding depth.
        repo_root = self._find_repo_root()
        if repo_root is not None:
            candidates.append(repo_root / "Dataset" / filename)

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return None

    @staticmethod
    def _find_repo_root() -> Path | None:
        """Walk up from this file to find the repository root (.git marker)."""
        p = Path(__file__).resolve()
        for parent in p.parents:
            if (parent / ".git").exists():
                return parent
        return None

    @staticmethod
    def _class_names_by_index(dataset_json: dict) -> list[str]:
        labels = dataset_json.get("labels", {})
        ordered = sorted(labels.items(), key=lambda kv: int(kv[1]))
        return [name for name, _ in ordered]

    def configure_optimizers(self):
        mod = self._get_trainable_model()

        optimizer_backbone = torch.optim.SGD(
            mod.backbone.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler_backbone = PolyLRScheduler(optimizer_backbone, self.initial_lr, self.num_epochs)

        self.optimizer_hyp_head = torch.optim.AdamW(mod.hyp_head.parameters(), lr=self.hyp_lr)
        self.optimizer_hyp_emb = torch.optim.AdamW(
            mod.label_emb.parameters(),
            lr=self.hyp_lr * self.hyp_text_lr_ratio,
        )
        self.lr_scheduler_hyp_head = PolyLRScheduler(self.optimizer_hyp_head, self.hyp_lr, self.num_epochs)
        self.lr_scheduler_hyp_emb = PolyLRScheduler(
            self.optimizer_hyp_emb,
            self.hyp_lr * self.hyp_text_lr_ratio,
            self.num_epochs,
        )

        return optimizer_backbone, lr_scheduler_backbone

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_hyp_head.zero_grad(set_to_none=True)
        self.optimizer_hyp_emb.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            seg_output, voxel_emb, label_emb = self.network(data)
            seg_loss = self.loss(seg_output, target)
            hyp_labels = self._target_for_hyp_loss(target)
            hyp_loss = self.hyp_loss(voxel_emb, hyp_labels, label_emb)
            total_loss = seg_loss + self.hyp_weight * hyp_loss

        model = self._get_trainable_model()

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            optimizers_with_grad = [
                opt for opt in (self.optimizer, self.optimizer_hyp_head, self.optimizer_hyp_emb)
                if self._optimizer_has_grad(opt)
            ]
            for opt in optimizers_with_grad:
                self.grad_scaler.unscale_(opt)

            clip_grad_norm_(model.backbone.parameters(), 12)
            if self._is_unfreeze_epoch():
                clip_grad_norm_(model.label_emb.parameters(), self.hyp_text_grad_clip)

            for opt in optimizers_with_grad:
                self.grad_scaler.step(opt)
            if optimizers_with_grad:
                self.grad_scaler.update()
        else:
            total_loss.backward()

            clip_grad_norm_(model.backbone.parameters(), 12)
            if self._is_unfreeze_epoch():
                clip_grad_norm_(model.label_emb.parameters(), self.hyp_text_grad_clip)

            self.optimizer.step()
            self.optimizer_hyp_head.step()
            if self._optimizer_has_grad(self.optimizer_hyp_emb):
                self.optimizer_hyp_emb.step()

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "seg_loss": seg_loss.detach().cpu().numpy(),
            "hyp_loss": hyp_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            seg_output, voxel_emb, label_emb = self.network(data)
            seg_loss = self.loss(seg_output, target)
            hyp_labels = self._target_for_hyp_loss(target)
            hyp_loss = self.hyp_loss(voxel_emb, hyp_labels, label_emb)
            total_loss = seg_loss + self.hyp_weight * hyp_loss

        output_for_metrics = seg_output[0] if self.enable_deep_supervision else seg_output
        target_for_metrics = target[0] if self.enable_deep_supervision else target

        axes = [0] + list(range(2, output_for_metrics.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_for_metrics) > 0.5).long()
        else:
            output_seg = output_for_metrics.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output_for_metrics.shape,
                device=output_for_metrics.device,
                dtype=torch.float16,
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_for_metrics != self.label_manager.ignore_label).float()
                target_for_metrics[target_for_metrics == self.label_manager.ignore_label] = 0
            else:
                if target_for_metrics.dtype == torch.bool:
                    mask = ~target_for_metrics[:, -1:]
                else:
                    mask = 1 - target_for_metrics[:, -1:]
                target_for_metrics = target_for_metrics[:, :-1]
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot,
            target_for_metrics,
            axes=axes,
            mask=mask,
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "seg_loss": seg_loss.detach().cpu().numpy(),
            "hyp_loss": hyp_loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def on_train_epoch_start(self):
        self.network.train()

        self.lr_scheduler.step(self.current_epoch)
        self.lr_scheduler_hyp_head.step(self.current_epoch)
        self.lr_scheduler_hyp_emb.step(self.current_epoch)

        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Backbone LR: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}, "
            f"HypHead LR: {np.round(self.optimizer_hyp_head.param_groups[0]['lr'], decimals=5)}, "
            f"LabelEmb LR: {np.round(self.optimizer_hyp_emb.param_groups[0]['lr'], decimals=5)}"
        )

        self._ensure_logging_key("lrs_hyp_head")
        self._ensure_logging_key("lrs_hyp_emb")
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)
        self.logger.log("lrs_hyp_head", self.optimizer_hyp_head.param_groups[0]["lr"], self.current_epoch)
        self.logger.log("lrs_hyp_emb", self.optimizer_hyp_emb.param_groups[0]["lr"], self.current_epoch)

        self.hyp_loss.set_epoch(self.current_epoch, self.num_epochs)

        mod = self._get_trainable_model()
        if self.current_epoch < self.hyp_freeze_epochs:
            mod.label_emb.tangent_embeddings.requires_grad_(False)
        else:
            mod.label_emb.tangent_embeddings.requires_grad_(True)

    def on_train_epoch_end(self, train_outputs: List[dict]):
        # Intentionally NOT calling super() — we additionally log seg_loss and
        # hyp_loss. Base class only logs total train_losses; we reproduce that
        # plus hyperbolic-specific metrics. Review this override if base class
        # on_train_epoch_end is updated.
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            seg_losses_tr = [None for _ in range(dist.get_world_size())]
            hyp_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs["loss"])
            dist.all_gather_object(seg_losses_tr, outputs["seg_loss"])
            dist.all_gather_object(hyp_losses_tr, outputs["hyp_loss"])
            loss_here = np.vstack(losses_tr).mean()
            seg_loss_here = np.vstack(seg_losses_tr).mean()
            hyp_loss_here = np.vstack(hyp_losses_tr).mean()
        else:
            loss_here = np.mean(outputs["loss"])
            seg_loss_here = np.mean(outputs["seg_loss"])
            hyp_loss_here = np.mean(outputs["hyp_loss"])

        self._ensure_logging_key("train_seg_losses")
        self._ensure_logging_key("train_hyp_losses")
        self.logger.log("train_losses", loss_here, self.current_epoch)
        self.logger.log("train_seg_losses", seg_loss_here, self.current_epoch)
        self.logger.log("train_hyp_losses", hyp_loss_here, self.current_epoch)

        self.print_to_log_file(f"train_seg_loss {np.round(seg_loss_here, decimals=4)}")
        self.print_to_log_file(f"train_hyp_loss {np.round(hyp_loss_here, decimals=4)}")

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        # Note: collate_outputs is called here and again inside super(). The
        # double call is minor overhead since validation is not the bottleneck.
        outputs_collated = collate_outputs(val_outputs)
        hyp_loss_here = np.mean(outputs_collated["hyp_loss"])

        if self.is_ddp:
            hyp_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(hyp_losses_val, outputs_collated["hyp_loss"])
            hyp_loss_here = np.vstack(hyp_losses_val).mean()

        self._ensure_logging_key("val_hyp_losses")
        self.logger.log("val_hyp_losses", hyp_loss_here, self.current_epoch)
        self.print_to_log_file(f"val_hyp_loss {np.round(hyp_loss_here, decimals=4)}")
        super().on_validation_epoch_end(val_outputs)

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self._get_trainable_model()
        mod.backbone.decoder.deep_supervision = enabled
        mod.hyperbolic_mode = enabled

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank != 0:
            return

        if self.disable_checkpointing:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")
            return

        mod = self._get_trainable_model()

        checkpoint = {
            "network_weights": mod.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "optimizer_hyp_head_state": self.optimizer_hyp_head.state_dict(),
            "optimizer_hyp_emb_state": self.optimizer_hyp_emb.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "lr_scheduler_hyp_head_state": self.lr_scheduler_hyp_head.state_dict(),
            "lr_scheduler_hyp_emb_state": self.lr_scheduler_hyp_emb.state_dict(),
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = checkpoint.get(
            "inference_allowed_mirroring_axes", self.inference_allowed_mirroring_axes
        )

        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None and checkpoint.get("grad_scaler_state") is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])

        if "optimizer_hyp_head_state" in checkpoint:
            self.optimizer_hyp_head.load_state_dict(checkpoint["optimizer_hyp_head_state"])
        if "optimizer_hyp_emb_state" in checkpoint:
            self.optimizer_hyp_emb.load_state_dict(checkpoint["optimizer_hyp_emb_state"])
        if "lr_scheduler_hyp_head_state" in checkpoint:
            self.lr_scheduler_hyp_head.load_state_dict(checkpoint["lr_scheduler_hyp_head_state"])
        if "lr_scheduler_hyp_emb_state" in checkpoint:
            self.lr_scheduler_hyp_emb.load_state_dict(checkpoint["lr_scheduler_hyp_emb_state"])

    def _target_for_hyp_loss(self, target):
        if isinstance(target, list):
            return target[0].squeeze(1).long()
        return target.squeeze(1).long()

    def _get_trainable_model(self) -> HyperBodyNet:
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def _is_unfreeze_epoch(self) -> bool:
        """True during a window of epochs after label_emb is first unfrozen.
        Applies conservative gradient clipping to prevent gradient spikes
        when the previously-frozen embeddings start receiving gradients."""
        return self.hyp_freeze_epochs <= self.current_epoch < self.hyp_freeze_epochs + 30

    @staticmethod
    def _optimizer_has_grad(optimizer: torch.optim.Optimizer) -> bool:
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    return True
        return False

    def load_pretrained_backbone(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("network_weights", checkpoint)

        remapped = {}
        for key, value in state.items():
            clean_key = key[7:] if key.startswith("module.") else key
            remapped[f"backbone.{clean_key}"] = value

        self._get_trainable_model().load_state_dict(remapped, strict=False)
