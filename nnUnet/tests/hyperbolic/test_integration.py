import json
from pathlib import Path

import torch


def _load_metadata() -> tuple[dict, dict]:
    nnunet_root = Path(__file__).resolve().parents[2]
    plans_path = nnunet_root / "nnUNet_data" / "nnUNet_preprocessed" / "Dataset501_HyperBody" / "nnUNetPlans.json"
    dataset_json_path = nnunet_root / "nnUNet_data" / "nnUNet_raw" / "Dataset501_HyperBody" / "dataset.json"
    return json.loads(plans_path.read_text()), json.loads(dataset_json_path.read_text())


def _make_initialized_trainer():
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
    trainer.hyp_freeze_epochs = 0
    trainer.initialize()
    trainer.current_epoch = 0
    trainer.on_train_epoch_start()
    return trainer


def _make_batch(trainer, batch_size: int = 1):
    data = torch.randn(batch_size, 1, 32, 16, 32)
    with torch.no_grad():
        seg_output, _, _ = trainer.network(data)

    targets = []
    for out in seg_output:
        spatial = out.shape[2:]
        target = torch.randint(0, trainer.label_manager.num_segmentation_heads, (batch_size, 1, *spatial))
        targets.append(target)

    return {"data": data, "target": targets}


def test_train_step_backward_reaches_backbone_and_hyperbolic_branches():
    trainer = _make_initialized_trainer()
    batch = _make_batch(trainer)

    model = trainer._get_trainable_model()
    backbone_param = next(model.backbone.parameters())
    hyp_head_param = next(model.hyp_head.parameters())
    label_emb_param = model.label_emb.tangent_embeddings

    before_backbone = backbone_param.detach().clone()
    before_hyp_head = hyp_head_param.detach().clone()
    before_label = label_emb_param.detach().clone()

    out = trainer.train_step(batch)

    assert set(out.keys()) == {"loss", "seg_loss", "hyp_loss"}
    assert float(out["loss"]) == float(out["loss"])
    assert float(out["hyp_loss"]) == float(out["hyp_loss"])

    assert backbone_param.grad is not None
    assert hyp_head_param.grad is not None
    assert label_emb_param.grad is not None
    assert torch.count_nonzero(backbone_param.grad).item() > 0
    assert torch.count_nonzero(hyp_head_param.grad).item() > 0
    assert torch.count_nonzero(label_emb_param.grad).item() > 0

    assert not torch.allclose(before_backbone, backbone_param.detach())
    assert not torch.allclose(before_hyp_head, hyp_head_param.detach())
    assert not torch.allclose(before_label, label_emb_param.detach())


def test_validation_step_returns_hyp_loss_and_dice_buffers():
    trainer = _make_initialized_trainer()
    batch = _make_batch(trainer)

    out = trainer.validation_step(batch)

    assert "loss" in out and "hyp_loss" in out
    assert "tp_hard" in out and "fp_hard" in out and "fn_hard" in out
    assert float(out["loss"]) == float(out["loss"])
    assert float(out["hyp_loss"]) == float(out["hyp_loss"])


def test_checkpoint_round_trip_after_one_train_step(tmp_path):
    trainer = _make_initialized_trainer()
    batch = _make_batch(trainer)
    trainer.train_step(batch)

    ckpt_file = tmp_path / "integration_hyperbody.pth"
    trainer.save_checkpoint(str(ckpt_file))

    from nnunetv2.training.nnUNetTrainer.variants.hyperbolic.nnUNetTrainerHyperBody import (
        nnUNetTrainerHyperBody,
    )

    plans, dataset_json = _load_metadata()
    reloaded = nnUNetTrainerHyperBody(
        plans=plans,
        configuration="3d_fullres",
        fold=0,
        dataset_json=dataset_json,
        device=torch.device("cpu"),
    )
    reloaded.num_epochs = 10
    reloaded.initialize()
    reloaded.load_checkpoint(str(ckpt_file))

    assert reloaded.optimizer_hyp_head is not None
    assert reloaded.optimizer_hyp_emb is not None
    assert reloaded.lr_scheduler_hyp_head is not None
    assert reloaded.lr_scheduler_hyp_emb is not None


def test_train_step_with_frozen_label_emb_does_not_raise():
    """S-4: When label_emb is frozen (hyp_freeze_epochs > current_epoch),
    optimizer_hyp_emb has no gradients. train_step must skip its step
    instead of raising a GradScaler assertion error."""
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
    trainer.hyp_freeze_epochs = 5  # freeze for first 5 epochs
    trainer.initialize()
    trainer.current_epoch = 2  # well within freeze window
    trainer.on_train_epoch_start()

    model = trainer._get_trainable_model()
    # Verify label_emb is actually frozen
    assert model.label_emb.tangent_embeddings.requires_grad is False

    batch = _make_batch(trainer)
    # This must not raise — the _optimizer_has_grad guard should skip
    # optimizer_hyp_emb.step() when label_emb has no gradients.
    out = trainer.train_step(batch)

    assert "loss" in out and "hyp_loss" in out
    # label_emb params should NOT have been updated
    assert model.label_emb.tangent_embeddings.grad is None
    # But backbone and hyp_head should still have gradients
    assert next(model.backbone.parameters()).grad is not None
    assert next(model.hyp_head.parameters()).grad is not None
