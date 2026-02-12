"""
TDD tests for evaluate_nnunet_predictions.py

Tests cover:
1. compute_dice_per_class: perfect match, no overlap, partial overlap, absent class
2. compute_mean_dice: mean excluding background
3. compute_group_dice: grouped by organ category
4. evaluate_single_sample: loads NIfTI and computes Dice
5. evaluate_single_sample_original_space: crops to grid_occ_size before computing Dice
6. generate_results_table / save_results: correct JSON output format
7. plot_per_class_dice: generates PNG bar chart
8. Real-data integration test (if data available)
"""
import os
import json
import tempfile

import numpy as np
import nibabel as nib
import pytest

# ---- Constants matching the design ----
TARGET_SHAPE = (144, 128, 268)
NUM_CLASSES = 70


# ============================================================
# Helpers: create synthetic NIfTI files and .npz for testing
# ============================================================

def make_nifti(data, path, affine=None):
    """Save a numpy array as a NIfTI file."""
    if affine is None:
        affine = np.diag([4.0, 4.0, 4.0, 1.0])
    img = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(img, path)


def make_synthetic_npz(path, grid_shape=(100, 90, 200), seed=42):
    """Create a synthetic .npz file with grid_occ_size."""
    rng = np.random.RandomState(seed)
    grid_world_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grid_voxel_size = np.array([4.0, 4.0, 4.0], dtype=np.float32)
    grid_world_max = grid_world_min + np.array(grid_shape, dtype=np.float32) * grid_voxel_size
    sensor_pc = np.zeros((100, 3), dtype=np.float32)
    voxel_labels = rng.randint(0, NUM_CLASSES, size=grid_shape, dtype=np.uint8)
    grid_occ_size = np.array(grid_shape, dtype=np.int32)

    np.savez(path,
             sensor_pc=sensor_pc,
             voxel_labels=voxel_labels,
             grid_world_min=grid_world_min,
             grid_world_max=grid_world_max,
             grid_voxel_size=grid_voxel_size,
             grid_occ_size=grid_occ_size)
    return voxel_labels


# ============================================================
# Tests for compute_dice_per_class
# ============================================================

class TestComputeDicePerClass:
    """Test per-class Dice computation."""

    def test_perfect_match(self):
        """When pred == gt, all present classes should have Dice = 1.0."""
        from evaluate_nnunet_predictions import compute_dice_per_class

        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[0:5, :, :] = 1
        gt[5:10, :, :] = 2
        pred = gt.copy()

        dice_dict = compute_dice_per_class(pred, gt, num_classes=3)
        # class 0 is not present (no background voxels in this volume)
        # class 1 and 2 should be 1.0
        assert dice_dict[1] == pytest.approx(1.0)
        assert dice_dict[2] == pytest.approx(1.0)

    def test_no_overlap(self):
        """When pred and gt have zero intersection for a class, Dice = 0.0."""
        from evaluate_nnunet_predictions import compute_dice_per_class

        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[:5, :, :] = 1

        pred = np.zeros((10, 10, 10), dtype=np.uint8)
        pred[5:, :, :] = 1  # completely non-overlapping

        dice_dict = compute_dice_per_class(pred, gt, num_classes=2)
        assert dice_dict[1] == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap should yield a Dice between 0 and 1."""
        from evaluate_nnunet_predictions import compute_dice_per_class

        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[:5, :, :] = 1  # 500 voxels

        pred = np.zeros((10, 10, 10), dtype=np.uint8)
        pred[:3, :, :] = 1  # 300 voxels, overlap = 300

        dice_dict = compute_dice_per_class(pred, gt, num_classes=2)
        # Dice = 2 * 300 / (500 + 300) = 600 / 800 = 0.75
        assert dice_dict[1] == pytest.approx(0.75)

    def test_class_absent_in_both(self):
        """If a class is absent in both pred and gt, Dice should be NaN."""
        from evaluate_nnunet_predictions import compute_dice_per_class

        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[:5, :, :] = 1
        pred = gt.copy()

        dice_dict = compute_dice_per_class(pred, gt, num_classes=5)
        # Classes 2, 3, 4 are absent in both
        assert np.isnan(dice_dict[2])
        assert np.isnan(dice_dict[3])
        assert np.isnan(dice_dict[4])

    def test_class_in_gt_only(self):
        """If a class is in gt but not pred, Dice = 0.0 (false negatives)."""
        from evaluate_nnunet_predictions import compute_dice_per_class

        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[:5, :, :] = 1
        gt[5:, :, :] = 2

        pred = np.zeros((10, 10, 10), dtype=np.uint8)
        pred[:5, :, :] = 1
        # pred has no class 2

        dice_dict = compute_dice_per_class(pred, gt, num_classes=3)
        assert dice_dict[1] == pytest.approx(1.0)
        assert dice_dict[2] == pytest.approx(0.0)

    def test_class_in_pred_only(self):
        """If a class is in pred but not gt, Dice = 0.0 (false positives)."""
        from evaluate_nnunet_predictions import compute_dice_per_class

        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[:5, :, :] = 1

        pred = np.zeros((10, 10, 10), dtype=np.uint8)
        pred[:5, :, :] = 1
        pred[5:, :, :] = 2  # class 2 only in pred

        dice_dict = compute_dice_per_class(pred, gt, num_classes=3)
        assert dice_dict[1] == pytest.approx(1.0)
        assert dice_dict[2] == pytest.approx(0.0)

    def test_all_70_classes(self):
        """Test with all 70 classes present."""
        from evaluate_nnunet_predictions import compute_dice_per_class

        rng = np.random.RandomState(42)
        gt = rng.randint(0, 70, size=(20, 20, 20), dtype=np.uint8)
        pred = gt.copy()  # perfect match

        dice_dict = compute_dice_per_class(pred, gt, num_classes=70)
        # All present classes should have Dice = 1.0
        for c in range(70):
            if np.any(gt == c):
                assert dice_dict[c] == pytest.approx(1.0), (
                    f"Class {c}: expected 1.0, got {dice_dict[c]}"
                )

    def test_vectorized_no_python_loop_over_voxels(self):
        """Ensure the function handles large arrays efficiently (vectorized)."""
        from evaluate_nnunet_predictions import compute_dice_per_class
        import time

        gt = np.zeros((144, 128, 268), dtype=np.uint8)
        gt[:72, :, :] = 1
        gt[72:, :, :] = 2
        pred = gt.copy()

        start = time.time()
        dice_dict = compute_dice_per_class(pred, gt, num_classes=3)
        elapsed = time.time() - start

        # Should complete in under 5 seconds for vectorized implementation
        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s (likely not vectorized)"
        assert dice_dict[1] == pytest.approx(1.0)
        assert dice_dict[2] == pytest.approx(1.0)

    def test_background_class_zero(self):
        """Background class 0 should also get a valid Dice score."""
        from evaluate_nnunet_predictions import compute_dice_per_class

        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[:5, :, :] = 1
        pred = gt.copy()

        dice_dict = compute_dice_per_class(pred, gt, num_classes=2)
        # class 0 occupies 500 voxels in both
        assert dice_dict[0] == pytest.approx(1.0)
        assert dice_dict[1] == pytest.approx(1.0)


# ============================================================
# Tests for compute_mean_dice
# ============================================================

class TestComputeMeanDice:
    """Test mean Dice computation."""

    def test_excludes_background(self):
        """Mean Dice should exclude class 0 by default."""
        from evaluate_nnunet_predictions import compute_mean_dice

        dice_dict = {0: 0.5, 1: 0.8, 2: 0.6}
        mean = compute_mean_dice(dice_dict, exclude_background=True)
        assert mean == pytest.approx(0.7)  # (0.8 + 0.6) / 2

    def test_includes_background(self):
        """With exclude_background=False, class 0 is included."""
        from evaluate_nnunet_predictions import compute_mean_dice

        dice_dict = {0: 0.5, 1: 0.8, 2: 0.6}
        mean = compute_mean_dice(dice_dict, exclude_background=False)
        assert mean == pytest.approx((0.5 + 0.8 + 0.6) / 3)

    def test_skips_nan(self):
        """NaN values (absent classes) should be excluded from the mean."""
        from evaluate_nnunet_predictions import compute_mean_dice

        dice_dict = {0: 0.5, 1: 0.8, 2: float('nan'), 3: 0.6}
        mean = compute_mean_dice(dice_dict, exclude_background=True)
        # Only class 1 and 3 contribute: (0.8 + 0.6) / 2 = 0.7
        assert mean == pytest.approx(0.7)

    def test_all_nan_returns_nan(self):
        """If all non-background classes are NaN, mean should be NaN."""
        from evaluate_nnunet_predictions import compute_mean_dice

        dice_dict = {0: 0.5, 1: float('nan'), 2: float('nan')}
        mean = compute_mean_dice(dice_dict, exclude_background=True)
        assert np.isnan(mean)

    def test_single_class(self):
        """With only one non-background class, mean equals that class Dice."""
        from evaluate_nnunet_predictions import compute_mean_dice

        dice_dict = {0: 0.9, 1: 0.85}
        mean = compute_mean_dice(dice_dict, exclude_background=True)
        assert mean == pytest.approx(0.85)


# ============================================================
# Tests for compute_group_dice
# ============================================================

class TestComputeGroupDice:
    """Test grouped Dice computation by organ category."""

    def test_all_groups_present(self):
        """All four groups should be computed."""
        from evaluate_nnunet_predictions import compute_group_dice

        # Create a dice_dict with all 70 classes
        dice_dict = {i: 0.8 for i in range(70)}

        group_dice = compute_group_dice(dice_dict)
        assert "organs" in group_dice
        assert "soft_tissue" in group_dice
        assert "bones" in group_dice
        assert "muscles" in group_dice

    def test_organ_group_composition(self):
        """Organs group should include classes 1-12."""
        from evaluate_nnunet_predictions import compute_group_dice, ORGAN_GROUPS

        # Set all organ classes to 0.9, everything else to 0.0
        dice_dict = {i: 0.0 for i in range(70)}
        for c in range(1, 13):
            dice_dict[c] = 0.9

        group_dice = compute_group_dice(dice_dict)
        assert group_dice["organs"] == pytest.approx(0.9)

    def test_muscle_group_composition(self):
        """Muscles group should include classes 60-69."""
        from evaluate_nnunet_predictions import compute_group_dice

        dice_dict = {i: 0.0 for i in range(70)}
        for c in range(60, 70):
            dice_dict[c] = 0.7

        group_dice = compute_group_dice(dice_dict)
        assert group_dice["muscles"] == pytest.approx(0.7)

    def test_nan_handling_in_groups(self):
        """NaN values in a group should be excluded from group mean."""
        from evaluate_nnunet_predictions import compute_group_dice

        dice_dict = {i: float('nan') for i in range(70)}
        # Only set a few organ classes
        dice_dict[1] = 0.8
        dice_dict[2] = 0.6

        group_dice = compute_group_dice(dice_dict)
        assert group_dice["organs"] == pytest.approx(0.7)

    def test_group_all_nan(self):
        """If all classes in a group are NaN, group mean should be NaN."""
        from evaluate_nnunet_predictions import compute_group_dice

        dice_dict = {i: float('nan') for i in range(70)}
        group_dice = compute_group_dice(dice_dict)
        assert np.isnan(group_dice["organs"])


# ============================================================
# Tests for evaluate_single_sample
# ============================================================

class TestEvaluateSingleSample:
    """Test single-sample evaluation from NIfTI files."""

    def test_perfect_prediction(self, tmp_path):
        """When prediction equals ground truth, mean Dice should be 1.0."""
        from evaluate_nnunet_predictions import evaluate_single_sample

        gt = np.zeros((20, 20, 20), dtype=np.uint8)
        gt[:10, :, :] = 1
        gt[10:, :, :] = 2

        pred_path = str(tmp_path / "pred.nii.gz")
        gt_path = str(tmp_path / "gt.nii.gz")
        make_nifti(gt, pred_path)
        make_nifti(gt, gt_path)

        dice_dict = evaluate_single_sample(pred_path, gt_path, num_classes=3)
        assert dice_dict[1] == pytest.approx(1.0)
        assert dice_dict[2] == pytest.approx(1.0)

    def test_wrong_prediction(self, tmp_path):
        """Completely wrong prediction should have Dice = 0.0 for each class."""
        from evaluate_nnunet_predictions import evaluate_single_sample

        gt = np.zeros((20, 20, 20), dtype=np.uint8)
        gt[:10, :, :] = 1

        pred = np.zeros((20, 20, 20), dtype=np.uint8)
        pred[10:, :, :] = 1  # no overlap

        pred_path = str(tmp_path / "pred.nii.gz")
        gt_path = str(tmp_path / "gt.nii.gz")
        make_nifti(pred, pred_path)
        make_nifti(gt, gt_path)

        dice_dict = evaluate_single_sample(pred_path, gt_path, num_classes=2)
        assert dice_dict[1] == pytest.approx(0.0)

    def test_returns_all_classes(self, tmp_path):
        """Should return Dice for all classes up to num_classes."""
        from evaluate_nnunet_predictions import evaluate_single_sample

        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[:5, :, :] = 1

        pred_path = str(tmp_path / "pred.nii.gz")
        gt_path = str(tmp_path / "gt.nii.gz")
        make_nifti(gt, pred_path)
        make_nifti(gt, gt_path)

        dice_dict = evaluate_single_sample(pred_path, gt_path, num_classes=70)
        assert len(dice_dict) == 70
        # Classes 2-69 not present -> NaN
        for c in range(2, 70):
            assert np.isnan(dice_dict[c])


# ============================================================
# Tests for evaluate_single_sample_original_space
# ============================================================

class TestEvaluateSingleSampleOriginalSpace:
    """Test evaluation with cropping to original grid_occ_size."""

    def test_crop_before_dice(self, tmp_path):
        """Should crop pred and gt to grid_occ_size before computing Dice."""
        from evaluate_nnunet_predictions import evaluate_single_sample_original_space

        original_shape = (10, 10, 10)
        padded_shape = (20, 20, 20)

        # GT: class 1 in original region, 0 elsewhere (padded)
        gt = np.zeros(padded_shape, dtype=np.uint8)
        gt[:10, :10, :10] = 1

        # Pred: class 1 in original region AND in padded region
        pred = np.ones(padded_shape, dtype=np.uint8)

        pred_path = str(tmp_path / "pred.nii.gz")
        gt_path = str(tmp_path / "gt.nii.gz")
        make_nifti(pred, pred_path)
        make_nifti(gt, gt_path)

        # Create npz with grid_occ_size
        npz_path = str(tmp_path / "sample.npz")
        np.savez(npz_path,
                 sensor_pc=np.zeros((1, 3), dtype=np.float32),
                 voxel_labels=np.ones(original_shape, dtype=np.uint8),
                 grid_world_min=np.zeros(3, dtype=np.float32),
                 grid_world_max=np.ones(3, dtype=np.float32) * 40.0,
                 grid_voxel_size=np.array([4.0, 4.0, 4.0], dtype=np.float32),
                 grid_occ_size=np.array(original_shape, dtype=np.int32))

        dice_dict = evaluate_single_sample_original_space(
            pred_path, gt_path, npz_path, num_classes=2)

        # In original space (10,10,10), pred is all 1 and gt is all 1 -> Dice(1) = 1.0
        # In padded space, pred has extra class 1 voxels -> Dice would be different
        assert dice_dict[1] == pytest.approx(1.0)

    def test_padded_vs_original_space_difference(self, tmp_path):
        """Padded evaluation should differ from original space when padding has content."""
        from evaluate_nnunet_predictions import (
            evaluate_single_sample, evaluate_single_sample_original_space
        )

        original_shape = (10, 10, 10)
        padded_shape = (20, 20, 20)

        # GT: class 1 in original region only
        gt = np.zeros(padded_shape, dtype=np.uint8)
        gt[:10, :10, :10] = 1

        # Pred: class 1 everywhere (wrong in padded region, but correct in original)
        pred = np.ones(padded_shape, dtype=np.uint8)

        pred_path = str(tmp_path / "pred.nii.gz")
        gt_path = str(tmp_path / "gt.nii.gz")
        make_nifti(pred, pred_path)
        make_nifti(gt, gt_path)

        npz_path = str(tmp_path / "sample.npz")
        np.savez(npz_path,
                 sensor_pc=np.zeros((1, 3), dtype=np.float32),
                 voxel_labels=np.ones(original_shape, dtype=np.uint8),
                 grid_world_min=np.zeros(3, dtype=np.float32),
                 grid_world_max=np.ones(3, dtype=np.float32) * 40.0,
                 grid_voxel_size=np.array([4.0, 4.0, 4.0], dtype=np.float32),
                 grid_occ_size=np.array(original_shape, dtype=np.int32))

        # Padded evaluation
        dice_padded = evaluate_single_sample(pred_path, gt_path, num_classes=2)
        # Original space evaluation
        dice_original = evaluate_single_sample_original_space(
            pred_path, gt_path, npz_path, num_classes=2)

        # In original space, class 1: perfect match -> 1.0
        assert dice_original[1] == pytest.approx(1.0)
        # In padded space, class 1: pred has extra voxels -> Dice < 1.0
        assert dice_padded[1] < 1.0


# ============================================================
# Tests for save_results / JSON output
# ============================================================

class TestSaveResults:
    """Test JSON output generation."""

    def test_json_output_format(self, tmp_path):
        """Output JSON should have per_case, mean_dice, per_class_mean, group_dice."""
        from evaluate_nnunet_predictions import save_results

        results = {
            "BDMAP_00000001": {i: 0.8 for i in range(70)},
            "BDMAP_00000002": {i: 0.7 for i in range(70)},
        }

        output_path = str(tmp_path / "results.json")
        save_results(results, output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)

        assert "per_case" in data
        assert "mean_dice" in data
        assert "per_class_mean" in data
        assert "group_dice" in data

    def test_per_case_structure(self, tmp_path):
        """per_case should contain per-class Dice for each sample."""
        from evaluate_nnunet_predictions import save_results

        results = {
            "BDMAP_00000001": {0: 0.9, 1: 0.8, 2: 0.7},
        }

        output_path = str(tmp_path / "results.json")
        save_results(results, output_path)

        with open(output_path) as f:
            data = json.load(f)

        case_data = data["per_case"]["BDMAP_00000001"]
        assert case_data["1"] == pytest.approx(0.8)  # JSON keys are strings

    def test_mean_dice_is_float(self, tmp_path):
        """mean_dice should be a single float value."""
        from evaluate_nnunet_predictions import save_results

        results = {
            "BDMAP_00000001": {i: 0.8 for i in range(70)},
        }

        output_path = str(tmp_path / "results.json")
        save_results(results, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert isinstance(data["mean_dice"], float)


# ============================================================
# Tests for plot_per_class_dice
# ============================================================

class TestPlotPerClassDice:
    """Test bar chart PNG generation."""

    def test_creates_png_file(self, tmp_path):
        """Should generate a PNG file at the given path."""
        from evaluate_nnunet_predictions import plot_per_class_dice

        mean_dice_dict = {i: 0.5 + 0.005 * i for i in range(70)}
        output_path = str(tmp_path / "dice_chart.png")

        plot_per_class_dice(mean_dice_dict, output_path)
        assert os.path.exists(output_path)
        # Check file is non-empty
        assert os.path.getsize(output_path) > 0

    def test_handles_nan_values(self, tmp_path):
        """Should handle NaN values gracefully (e.g., show them as 0 or skip)."""
        from evaluate_nnunet_predictions import plot_per_class_dice

        mean_dice_dict = {i: float('nan') if i % 3 == 0 else 0.8 for i in range(70)}
        output_path = str(tmp_path / "dice_chart_nan.png")

        # Should not raise
        plot_per_class_dice(mean_dice_dict, output_path)
        assert os.path.exists(output_path)


# ============================================================
# Tests for evaluate_folder
# ============================================================

class TestEvaluateFolder:
    """Test full folder evaluation."""

    def test_evaluates_all_matching_files(self, tmp_path):
        """Should evaluate all files that exist in both pred and gt dirs."""
        from evaluate_nnunet_predictions import evaluate_folder

        pred_dir = tmp_path / "predictions"
        gt_dir = tmp_path / "labelsTs"
        pred_dir.mkdir()
        gt_dir.mkdir()

        # Create two matching pairs
        for case_id in ["BDMAP_00000001", "BDMAP_00000002"]:
            gt = np.zeros((10, 10, 10), dtype=np.uint8)
            gt[:5, :, :] = 1
            make_nifti(gt, str(pred_dir / f"{case_id}.nii.gz"))
            make_nifti(gt, str(gt_dir / f"{case_id}.nii.gz"))

        results = evaluate_folder(str(pred_dir), str(gt_dir), num_classes=2,
                                  num_workers=1)

        assert len(results) == 2
        assert "BDMAP_00000001" in results
        assert "BDMAP_00000002" in results

    def test_skips_unmatched_files(self, tmp_path):
        """Files only in pred or only in gt should be skipped."""
        from evaluate_nnunet_predictions import evaluate_folder

        pred_dir = tmp_path / "predictions"
        gt_dir = tmp_path / "labelsTs"
        pred_dir.mkdir()
        gt_dir.mkdir()

        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[:5, :, :] = 1

        # Matching pair
        make_nifti(gt, str(pred_dir / "BDMAP_00000001.nii.gz"))
        make_nifti(gt, str(gt_dir / "BDMAP_00000001.nii.gz"))

        # Only in pred
        make_nifti(gt, str(pred_dir / "BDMAP_00000099.nii.gz"))

        # Only in gt
        make_nifti(gt, str(gt_dir / "BDMAP_00000088.nii.gz"))

        results = evaluate_folder(str(pred_dir), str(gt_dir), num_classes=2,
                                  num_workers=1)
        assert len(results) == 1
        assert "BDMAP_00000001" in results


# ============================================================
# Integration test on real data (if available)
# ============================================================

class TestRealDataIntegration:
    """Integration test using real labelsTs files."""

    LABELS_TS_DIR = "/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_raw/Dataset501_HyperBody/labelsTs"
    NPZ_DIR = "/home/comp/csrkzhu/code/Compare/nnUNet/Dataset/voxel_data"
    SPLIT_JSON = "/home/comp/csrkzhu/code/Compare/nnUNet/Dataset/dataset_split.json"

    @pytest.mark.skipif(
        not os.path.isdir(
            "/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_raw/Dataset501_HyperBody/labelsTs"),
        reason="labelsTs directory not available"
    )
    def test_real_gt_self_evaluation(self, tmp_path):
        """Evaluate GT against itself: all Dice should be 1.0."""
        from evaluate_nnunet_predictions import evaluate_single_sample

        # Pick the first available label file
        label_files = sorted(os.listdir(self.LABELS_TS_DIR))
        if not label_files:
            pytest.skip("No label files found")

        gt_path = os.path.join(self.LABELS_TS_DIR, label_files[0])

        # Evaluate GT against itself
        dice_dict = evaluate_single_sample(gt_path, gt_path, num_classes=70)

        # All present classes should have Dice = 1.0
        for c, d in dice_dict.items():
            if not np.isnan(d):
                assert d == pytest.approx(1.0), (
                    f"Class {c}: expected 1.0, got {d}"
                )

        # Print summary
        present = sum(1 for d in dice_dict.values() if not np.isnan(d))
        print(f"\n--- Real GT self-evaluation: {label_files[0]} ---")
        print(f"  Present classes: {present}/{NUM_CLASSES}")

    @pytest.mark.skipif(
        not (os.path.isdir(
            "/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_raw/Dataset501_HyperBody/labelsTs")
            and os.path.isdir(
                "/home/comp/csrkzhu/code/Compare/nnUNet/Dataset/voxel_data")),
        reason="labelsTs and/or npz data not available"
    )
    def test_real_original_space_self_evaluation(self, tmp_path):
        """Evaluate GT against itself in original space."""
        from evaluate_nnunet_predictions import evaluate_single_sample_original_space

        with open(self.SPLIT_JSON) as f:
            split = json.load(f)

        test_file = split["test"][0]
        case_id = test_file.replace(".npz", "")

        gt_path = os.path.join(self.LABELS_TS_DIR, f"{case_id}.nii.gz")
        npz_path = os.path.join(self.NPZ_DIR, test_file)

        if not os.path.exists(gt_path) or not os.path.exists(npz_path):
            pytest.skip(f"Files not available: {gt_path} or {npz_path}")

        dice_dict = evaluate_single_sample_original_space(
            gt_path, gt_path, npz_path, num_classes=70)

        for c, d in dice_dict.items():
            if not np.isnan(d):
                assert d == pytest.approx(1.0), (
                    f"Class {c}: expected 1.0, got {d}"
                )

        present = sum(1 for d in dice_dict.values() if not np.isnan(d))
        print(f"\n--- Real original-space self-evaluation: {case_id} ---")
        print(f"  Present classes: {present}/{NUM_CLASSES}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
