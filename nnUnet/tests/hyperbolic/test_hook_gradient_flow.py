import torch

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


ARCH_CLASS_NAME = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
ARCH_INIT_KWARGS = {
    "n_stages": 5,
    "features_per_stage": [32, 64, 128, 256, 320],
    "conv_op": "torch.nn.modules.conv.Conv3d",
    "kernel_sizes": [[3, 3, 3]] * 5,
    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "n_conv_per_stage": [2, 2, 2, 2, 2],
    "n_conv_per_stage_decoder": [2, 2, 2, 2],
    "conv_bias": True,
    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
    "dropout_op": None,
    "dropout_op_kwargs": None,
    "nonlin": "torch.nn.LeakyReLU",
    "nonlin_kwargs": {"inplace": True},
}
ARCH_INIT_KWARGS_REQ_IMPORT = ["conv_op", "norm_op", "dropout_op", "nonlin"]


def _build_plain_conv_unet() -> torch.nn.Module:
    return get_network_from_plans(
        ARCH_CLASS_NAME,
        ARCH_INIT_KWARGS,
        ARCH_INIT_KWARGS_REQ_IMPORT,
        input_channels=1,
        output_channels=70,
        allow_init=True,
        deep_supervision=True,
    )


def _assert_hook_gradient_flow(model: torch.nn.Module) -> None:
    model.train()
    captured = {}

    def _capture(_module, _inputs, output):
        captured["features"] = output

    hook = model.decoder.stages[-1].register_forward_hook(_capture)
    x = torch.randn(1, 1, 32, 16, 32)
    _ = model(x)
    assert "features" in captured
    assert tuple(captured["features"].shape) == (1, 32, 32, 16, 32)

    captured["features"].sum().backward()

    encoder_grad_sums = [
        p.grad.abs().sum().item()
        for p in model.encoder.parameters()
        if p.grad is not None
    ]
    assert encoder_grad_sums
    assert any(grad_sum > 0 for grad_sum in encoder_grad_sums)
    hook.remove()


def test_hook_captures_decoder_features_and_backpropagates_to_encoder():
    model = _build_plain_conv_unet()
    _assert_hook_gradient_flow(model)


def test_hook_gradient_flow_still_works_with_torch_compile():
    if not hasattr(torch, "compile"):
        return

    compiled_model = torch.compile(_build_plain_conv_unet())
    _assert_hook_gradient_flow(compiled_model)
