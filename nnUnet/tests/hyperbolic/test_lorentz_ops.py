import torch

from nnunetv2.training.nnUNetTrainer.variants.hyperbolic.lorentz_ops import (
    distance_to_origin,
    exp_map0,
    log_map0,
    pairwise_dist,
    pointwise_dist,
)


def test_exp_log_round_trip_keeps_shape_and_is_close():
    v = torch.randn(8, 32) * 0.2
    x = exp_map0(v, curv=1.0)
    v_recon = log_map0(x, curv=1.0)

    assert x.shape == v.shape
    assert v_recon.shape == v.shape
    assert torch.allclose(v, v_recon, atol=2e-3, rtol=2e-3)


def test_pairwise_and_pointwise_distance_shapes_and_consistency():
    a = exp_map0(torch.randn(5, 16) * 0.1)
    b = exp_map0(torch.randn(7, 16) * 0.1)

    all_pairs = pairwise_dist(a, b)
    assert all_pairs.shape == (5, 7)

    i = torch.tensor([0, 1, 2])
    j = torch.tensor([2, 3, 4])
    d_point = pointwise_dist(a[i], b[j])
    d_from_pairwise = all_pairs[i, j]
    assert d_point.shape == (3,)
    assert torch.allclose(d_point, d_from_pairwise, atol=1e-5, rtol=1e-5)


def test_distances_are_finite_and_non_negative():
    x = exp_map0(torch.randn(10, 24) * 0.2)
    y = exp_map0(torch.randn(10, 24) * 0.2)

    d_xy = pointwise_dist(x, y)
    d_x0 = distance_to_origin(x)

    assert torch.isfinite(d_xy).all()
    assert torch.isfinite(d_x0).all()
    assert (d_xy >= 0).all()
    assert (d_x0 >= 0).all()
