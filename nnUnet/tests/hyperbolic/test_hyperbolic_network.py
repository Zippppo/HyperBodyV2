import torch

from nnunetv2.training.nnUNetTrainer.variants.hyperbolic.hyperbolic_network import HyperBodyNet
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def _build_backbone(num_classes: int = 70) -> torch.nn.Module:
    return get_network_from_plans(
        "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        {
            "n_stages": 5,
            "features_per_stage": [8, 16, 32, 64, 64],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[3, 3, 3]] * 5,
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "n_conv_per_stage": [1, 1, 1, 1, 1],
            "n_conv_per_stage_decoder": [1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        ["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=1,
        output_channels=num_classes,
        allow_init=True,
        deep_supervision=True,
    )


def test_hyperbolic_forward_returns_seg_voxel_and_label_embeddings():
    backbone = _build_backbone(num_classes=70)
    model = HyperBodyNet(backbone=backbone, num_classes=70)

    seg_output, voxel_emb, label_emb = model(torch.randn(1, 1, 32, 16, 32))

    assert isinstance(seg_output, list)
    assert tuple(seg_output[0].shape) == (1, 70, 32, 16, 32)
    assert tuple(voxel_emb.shape) == (1, 8, 32, 16, 32)
    assert tuple(label_emb.shape) == (70, 8)
    assert model._decoder_features is None


def test_inference_mode_returns_backbone_output_only():
    backbone = _build_backbone(num_classes=5)
    model = HyperBodyNet(backbone=backbone, num_classes=5)
    model.hyperbolic_mode = False

    seg_output = model(torch.randn(1, 1, 32, 16, 32))

    assert isinstance(seg_output, list)
    assert tuple(seg_output[0].shape) == (1, 5, 32, 16, 32)
    assert model._decoder_features is None


def test_state_dict_contains_backbone_and_hyperbolic_params():
    backbone = _build_backbone(num_classes=3)
    model = HyperBodyNet(backbone=backbone, num_classes=3)
    state_dict = model.state_dict()

    assert any(k.startswith("backbone.") for k in state_dict)
    assert any(k.startswith("hyp_head.") for k in state_dict)
    assert any(k.startswith("label_emb.") for k in state_dict)


def test_compute_conv_feature_map_size_delegates_to_backbone():
    backbone = _build_backbone(num_classes=3)
    model = HyperBodyNet(backbone=backbone, num_classes=3)

    expected = backbone.compute_conv_feature_map_size((32, 16, 32))
    actual = model.compute_conv_feature_map_size((32, 16, 32))

    assert actual == expected
