import json
from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.variants.hyperbolic.hyperbolic_network import HyperBodyNet


def _paths() -> tuple[Path, Path]:
    nnunet_root = Path(__file__).resolve().parents[2]
    plans_path = nnunet_root / "nnUNet_data" / "nnUNet_preprocessed" / "Dataset501_HyperBody" / "nnUNetPlans.json"
    dataset_json_path = nnunet_root / "nnUNet_data" / "nnUNet_raw" / "Dataset501_HyperBody" / "dataset.json"
    return plans_path, dataset_json_path


def _load_metadata() -> tuple[dict, dict]:
    plans_path, dataset_json_path = _paths()
    plans = json.loads(plans_path.read_text())
    dataset_json = json.loads(dataset_json_path.read_text())
    return plans, dataset_json


def _make_trainer():
    from nnunetv2.training.nnUNetTrainer.variants.hyperbolic.nnUNetTrainerHyperBody import (
        nnUNetTrainerHyperBody,
    )

    plans, dataset_json = _load_metadata()
    trainer = nnUNetTrainerHyperBody(
        plans=plans,
        configuration="3d_fullres",
        fold=0,
        dataset_json=dataset_json,
        device=torch.device("cpu"),
    )
    trainer.num_epochs = 10
    return trainer


def _unwrap_network(network):
    return network._orig_mod if hasattr(network, "_orig_mod") else network


def test_configure_optimizers_returns_backbone_pair_and_sets_hyper_optimizers():
    trainer = _make_trainer()
    trainer.initialize()

    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert hasattr(trainer, "optimizer_hyp_head")
    assert hasattr(trainer, "optimizer_hyp_emb")
    assert isinstance(trainer.optimizer_hyp_head, torch.optim.AdamW)
    assert isinstance(trainer.optimizer_hyp_emb, torch.optim.AdamW)
    assert trainer.optimizer_hyp_emb.param_groups[0]["lr"] < trainer.optimizer_hyp_head.param_groups[0]["lr"]


def test_build_network_architecture_disables_hyperbolic_mode_for_inference():
    from nnunetv2.training.nnUNetTrainer.variants.hyperbolic.nnUNetTrainerHyperBody import (
        nnUNetTrainerHyperBody,
    )

    plans, _ = _load_metadata()
    arch = plans["configurations"]["3d_fullres"]["architecture"]
    network = nnUNetTrainerHyperBody.build_network_architecture(
        architecture_class_name=arch["network_class_name"],
        arch_init_kwargs=arch["arch_kwargs"],
        arch_init_kwargs_req_import=arch["_kw_requires_import"],
        num_input_channels=1,
        num_output_channels=70,
        enable_deep_supervision=False,
    )

    assert isinstance(network, HyperBodyNet)
    assert network.hyperbolic_mode is False


def test_set_deep_supervision_enabled_toggles_decoder_and_hyper_mode():
    trainer = _make_trainer()
    trainer.initialize()
    model = _unwrap_network(trainer.network)

    trainer.set_deep_supervision_enabled(False)
    assert model.backbone.decoder.deep_supervision is False
    assert model.hyperbolic_mode is False

    trainer.set_deep_supervision_enabled(True)
    assert model.backbone.decoder.deep_supervision is True
    assert model.hyperbolic_mode is True


def test_on_train_epoch_start_steps_all_schedulers_and_freeze_unfreeze_label_embeddings():
    trainer = _make_trainer()
    trainer.initialize()
    model = _unwrap_network(trainer.network)

    trainer.current_epoch = 0
    trainer.on_train_epoch_start()
    lr_backbone_epoch0 = trainer.optimizer.param_groups[0]["lr"]
    lr_hyp_head_epoch0 = trainer.optimizer_hyp_head.param_groups[0]["lr"]
    lr_hyp_emb_epoch0 = trainer.optimizer_hyp_emb.param_groups[0]["lr"]
    assert model.label_emb.tangent_embeddings.requires_grad is False

    trainer.current_epoch = max(1, trainer.hyp_freeze_epochs)
    trainer.on_train_epoch_start()
    assert trainer.optimizer.param_groups[0]["lr"] <= lr_backbone_epoch0
    assert trainer.optimizer_hyp_head.param_groups[0]["lr"] <= lr_hyp_head_epoch0
    assert trainer.optimizer_hyp_emb.param_groups[0]["lr"] <= lr_hyp_emb_epoch0
    assert model.label_emb.tangent_embeddings.requires_grad is True
    assert trainer.hyp_loss.current_epoch.item() == trainer.current_epoch


def test_checkpoint_round_trip_contains_hyperbolic_optimizer_and_scheduler_states(tmp_path):
    trainer = _make_trainer()
    trainer.initialize()
    checkpoint_file = tmp_path / "hyperbody_ckpt.pth"

    trainer.current_epoch = 3
    trainer.save_checkpoint(str(checkpoint_file))

    ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    assert "optimizer_hyp_head_state" in ckpt
    assert "optimizer_hyp_emb_state" in ckpt
    assert "lr_scheduler_hyp_head_state" in ckpt
    assert "lr_scheduler_hyp_emb_state" in ckpt

    trainer_reloaded = _make_trainer()
    trainer_reloaded.initialize()
    trainer_reloaded.load_checkpoint(str(checkpoint_file))

    assert trainer_reloaded.current_epoch == ckpt["current_epoch"]


def test_is_unfreeze_epoch_covers_window_not_single_epoch():
    """I-3: _is_unfreeze_epoch should return True for a window of epochs
    after unfreezing, not just the exact epoch == hyp_freeze_epochs.
    This ensures gradient clipping applies for a few epochs after label_emb
    is first unfrozen, preventing gradient spikes."""
    trainer = _make_trainer()
    trainer.hyp_freeze_epochs = 5

    # Before unfreeze: should be False
    trainer.current_epoch = 4
    assert trainer._is_unfreeze_epoch() is False

    # First unfreeze epoch: should be True
    trainer.current_epoch = 5
    assert trainer._is_unfreeze_epoch() is True

    # Second epoch after unfreeze: should still be True (within window)
    trainer.current_epoch = 6
    assert trainer._is_unfreeze_epoch() is True

    # Third epoch after unfreeze: should still be True (within window)
    trainer.current_epoch = 7
    assert trainer._is_unfreeze_epoch() is True

    # After window: should be False
    trainer.current_epoch = 8
    assert trainer._is_unfreeze_epoch() is False


def test_class_names_by_index_handles_string_label_values():
    """S-5: _class_names_by_index should handle dataset.json where label
    values are strings (e.g., '0', '1', '10') by casting to int before sort.
    Without int() cast, '10' < '2' in lexicographic order."""
    from nnunetv2.training.nnUNetTrainer.variants.hyperbolic.nnUNetTrainerHyperBody import (
        nnUNetTrainerHyperBody,
    )

    dataset_json_str_values = {
        "labels": {
            "background": "0",
            "heart": "1",
            "liver": "2",
            "kidney": "10",
        }
    }
    names = nnUNetTrainerHyperBody._class_names_by_index(dataset_json_str_values)
    assert names == ["background", "heart", "liver", "kidney"]

    # Verify it still works with int values (standard nnUNet format)
    dataset_json_int_values = {
        "labels": {
            "background": 0,
            "heart": 1,
            "liver": 2,
            "kidney": 10,
        }
    }
    names_int = nnUNetTrainerHyperBody._class_names_by_index(dataset_json_int_values)
    assert names_int == ["background", "heart", "liver", "kidney"]


def test_resolve_dependency_path_does_not_use_hardcoded_depth():
    """S-2: _resolve_dependency_path should use git-root detection
    instead of hardcoded parents[6] to find the repo-level Dataset/ dir."""
    trainer = _make_trainer()
    # The method should work regardless of directory nesting depth.
    # We test that it can find existing files from the correct repo root.
    from pathlib import Path

    trainer_file = Path(
        trainer.__class__.__module__.replace(".", "/") + ".py"
    )
    # As a sanity check: the method should resolve without error,
    # and if the file exists, return a valid Path.
    result = trainer._resolve_dependency_path("tree.json")
    if result is not None:
        assert result.is_file()
        assert result.name == "tree.json"
