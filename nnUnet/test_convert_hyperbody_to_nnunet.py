"""
TDD tests for convert_hyperbody_to_nnunet.py

Tests cover:
1. Single sample conversion correctness (occupancy grid + labels)
2. NIfTI format, affine, spacing verification
3. Edge case: sample exceeding target bounds (clipping)
4. dataset.json generation
5. File naming conventions
6. Full pipeline integration test on real data
"""
import os
import json
import tempfile
import shutil

import numpy as np
import nibabel as nib
import pytest

# ---- Constants matching the design doc ----
TARGET_SHAPE = (144, 128, 268)
VOXEL_SIZE = np.array([4.0, 4.0, 4.0], dtype=np.float32)
NUM_CLASSES = 70


# ============================================================
# Helpers: create synthetic .npz samples for testing
# ============================================================

def make_synthetic_npz(path, grid_shape=(100, 90, 200), n_points=5000,
                       grid_world_min=None, seed=42):
    """Create a synthetic .npz file mimicking HyperBody format."""
    rng = np.random.RandomState(seed)
    if grid_world_min is None:
        grid_world_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    grid_voxel_size = np.array([4.0, 4.0, 4.0], dtype=np.float32)
    grid_world_max = grid_world_min + np.array(grid_shape, dtype=np.float32) * grid_voxel_size

    # Generate random points within the grid world bounds
    sensor_pc = np.zeros((n_points, 3), dtype=np.float32)
    for d in range(3):
        sensor_pc[:, d] = rng.uniform(grid_world_min[d], grid_world_max[d], n_points)

    # Generate random voxel labels (0-69)
    voxel_labels = rng.randint(0, NUM_CLASSES, size=grid_shape, dtype=np.uint8)

    grid_occ_size = np.array(grid_shape, dtype=np.int32)

    np.savez(path,
             sensor_pc=sensor_pc,
             voxel_labels=voxel_labels,
             grid_world_min=grid_world_min,
             grid_world_max=grid_world_max,
             grid_voxel_size=grid_voxel_size,
             grid_occ_size=grid_occ_size)
    return sensor_pc, voxel_labels, grid_world_min


def make_oversized_npz(path, seed=123):
    """Create a sample where Y=129 exceeds target Y=128 (BDMAP_00002911 case)."""
    return make_synthetic_npz(path, grid_shape=(100, 129, 200), seed=seed)


# ============================================================
# Test fixtures
# ============================================================

@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with source and output directories."""
    src_dir = tmp_path / "voxel_data"
    src_dir.mkdir()
    out_dir = tmp_path / "nnUNet_raw" / "Dataset501_HyperBody"
    return src_dir, out_dir, tmp_path


# ============================================================
# Unit Tests for conversion functions
# ============================================================

class TestBuildOccupancyGrid:
    """Test occupancy grid construction from point cloud."""

    def test_shape(self):
        from convert_hyperbody_to_nnunet import build_occupancy_grid
        sensor_pc = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        grid_world_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size = np.array([4.0, 4.0, 4.0], dtype=np.float32)

        occ = build_occupancy_grid(sensor_pc, grid_world_min, voxel_size, TARGET_SHAPE)
        assert occ.shape == TARGET_SHAPE, f"Expected {TARGET_SHAPE}, got {occ.shape}"
        assert occ.dtype == np.float32

    def test_single_point_placement(self):
        """A single point at known coordinates should fill exactly one voxel."""
        from convert_hyperbody_to_nnunet import build_occupancy_grid
        grid_world_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        # Point at (6.0, 10.0, 14.0) -> voxel index floor([1.5, 2.5, 3.5]) = (1, 2, 3)
        sensor_pc = np.array([[6.0, 10.0, 14.0]], dtype=np.float32)

        occ = build_occupancy_grid(sensor_pc, grid_world_min, voxel_size, TARGET_SHAPE)
        assert occ[1, 2, 3] == 1.0
        assert occ.sum() == 1.0, "Only one voxel should be occupied"

    def test_duplicate_points(self):
        """Multiple points in the same voxel should still give value 1.0."""
        from convert_hyperbody_to_nnunet import build_occupancy_grid
        grid_world_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        sensor_pc = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],  # same voxel (0, 0, 0)
            [5.0, 5.0, 5.0],  # voxel (1, 1, 1)
        ], dtype=np.float32)

        occ = build_occupancy_grid(sensor_pc, grid_world_min, voxel_size, TARGET_SHAPE)
        assert occ[0, 0, 0] == 1.0
        assert occ[1, 1, 1] == 1.0
        assert occ.sum() == 2.0

    def test_clipping_out_of_bounds(self):
        """Points outside target bounds should be clipped, not cause errors."""
        from convert_hyperbody_to_nnunet import build_occupancy_grid
        grid_world_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        # Point far beyond target bounds
        sensor_pc = np.array([[9999.0, 9999.0, 9999.0]], dtype=np.float32)

        occ = build_occupancy_grid(sensor_pc, grid_world_min, voxel_size, TARGET_SHAPE)
        assert occ.shape == TARGET_SHAPE
        # Clipped to the corner voxel
        assert occ[TARGET_SHAPE[0]-1, TARGET_SHAPE[1]-1, TARGET_SHAPE[2]-1] == 1.0
        assert occ.sum() == 1.0

    def test_negative_indices_clipped(self):
        """Points before grid_world_min should be clipped to 0."""
        from convert_hyperbody_to_nnunet import build_occupancy_grid
        grid_world_min = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        voxel_size = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        # Point before the grid origin
        sensor_pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        occ = build_occupancy_grid(sensor_pc, grid_world_min, voxel_size, TARGET_SHAPE)
        assert occ[0, 0, 0] == 1.0
        assert occ.sum() == 1.0


class TestPadLabels:
    """Test label padding/clipping to target shape."""

    def test_smaller_than_target(self):
        """Labels smaller than target should be corner-padded with zeros."""
        from convert_hyperbody_to_nnunet import pad_labels
        voxel_labels = np.ones((80, 60, 100), dtype=np.uint8) * 5
        padded = pad_labels(voxel_labels, TARGET_SHAPE)

        assert padded.shape == TARGET_SHAPE
        assert padded.dtype == np.uint8
        # Original region preserved
        assert np.all(padded[:80, :60, :100] == 5)
        # Padded region is zeros
        assert np.all(padded[80:, :, :] == 0)
        assert np.all(padded[:, 60:, :] == 0)
        assert np.all(padded[:, :, 100:] == 0)

    def test_exact_target_size(self):
        """Labels matching target exactly should be copied as-is."""
        from convert_hyperbody_to_nnunet import pad_labels
        voxel_labels = np.ones(TARGET_SHAPE, dtype=np.uint8) * 3
        padded = pad_labels(voxel_labels, TARGET_SHAPE)

        assert padded.shape == TARGET_SHAPE
        assert np.all(padded == 3)

    def test_larger_than_target_clipped(self):
        """Labels exceeding target should be clipped (BDMAP_00002911 case: Y=129)."""
        from convert_hyperbody_to_nnunet import pad_labels
        voxel_labels = np.ones((100, 129, 200), dtype=np.uint8) * 7
        padded = pad_labels(voxel_labels, TARGET_SHAPE)

        assert padded.shape == TARGET_SHAPE
        # Only first 128 in Y are kept (clipped from 129)
        assert np.all(padded[:100, :128, :200] == 7)
        # Padded region in X beyond 100
        assert np.all(padded[100:, :, :] == 0)

    def test_preserves_all_classes(self):
        """All 70 class values (0-69) should be preserved after padding."""
        from convert_hyperbody_to_nnunet import pad_labels
        rng = np.random.RandomState(42)
        voxel_labels = rng.randint(0, 70, size=(80, 60, 100), dtype=np.uint8)
        padded = pad_labels(voxel_labels, TARGET_SHAPE)

        original_classes = set(np.unique(voxel_labels))
        padded_classes = set(np.unique(padded[:80, :60, :100]))
        assert original_classes == padded_classes


class TestCreateNiftiAffine:
    """Test NIfTI affine matrix construction."""

    def test_affine_diagonal(self):
        from convert_hyperbody_to_nnunet import create_nifti_affine
        grid_world_min = np.array([-100.0, -50.0, 0.0], dtype=np.float32)
        affine = create_nifti_affine(grid_world_min, VOXEL_SIZE)

        assert affine.shape == (4, 4)
        # Diagonal = voxel sizes
        assert affine[0, 0] == 4.0
        assert affine[1, 1] == 4.0
        assert affine[2, 2] == 4.0
        assert affine[3, 3] == 1.0
        # Origin = grid_world_min
        np.testing.assert_array_almost_equal(affine[:3, 3], grid_world_min)


class TestConvertSingleSample:
    """Integration test: convert a single .npz to NIfTI files."""

    def test_convert_produces_correct_files(self, tmp_workspace):
        from convert_hyperbody_to_nnunet import convert_single_sample
        src_dir, out_dir, tmp_path = tmp_workspace

        npz_path = str(src_dir / "BDMAP_00000001.npz")
        sensor_pc, voxel_labels, grid_world_min = make_synthetic_npz(npz_path)

        img_dir = out_dir / "imagesTr"
        lbl_dir = out_dir / "labelsTr"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        convert_single_sample(npz_path, str(img_dir), str(lbl_dir))

        # Check files exist with correct naming
        img_path = img_dir / "BDMAP_00000001_0000.nii.gz"
        lbl_path = lbl_dir / "BDMAP_00000001.nii.gz"
        assert img_path.exists(), f"Image file not found: {img_path}"
        assert lbl_path.exists(), f"Label file not found: {lbl_path}"

    def test_nifti_content_correctness(self, tmp_workspace):
        from convert_hyperbody_to_nnunet import convert_single_sample
        src_dir, out_dir, tmp_path = tmp_workspace

        npz_path = str(src_dir / "BDMAP_00000001.npz")
        sensor_pc, voxel_labels, grid_world_min = make_synthetic_npz(
            npz_path, grid_shape=(80, 60, 100), n_points=100, seed=99)

        img_dir = out_dir / "imagesTr"
        lbl_dir = out_dir / "labelsTr"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        convert_single_sample(npz_path, str(img_dir), str(lbl_dir))

        # Load and verify image
        img_nii = nib.load(str(img_dir / "BDMAP_00000001_0000.nii.gz"))
        img_data = img_nii.get_fdata()
        assert img_data.shape == TARGET_SHAPE, f"Image shape: {img_data.shape}"

        # Verify affine
        affine = img_nii.affine
        np.testing.assert_array_almost_equal(np.diag(affine), [4.0, 4.0, 4.0, 1.0])
        np.testing.assert_array_almost_equal(affine[:3, 3], grid_world_min)

        # Verify binary occupancy
        unique_vals = np.unique(img_data)
        assert set(unique_vals).issubset({0.0, 1.0}), f"Non-binary values: {unique_vals}"
        assert img_data.sum() > 0, "Occupancy grid should have non-zero voxels"

        # Load and verify label
        lbl_nii = nib.load(str(lbl_dir / "BDMAP_00000001.nii.gz"))
        lbl_data = lbl_nii.get_fdata().astype(np.uint8)
        assert lbl_data.shape == TARGET_SHAPE, f"Label shape: {lbl_data.shape}"

        # Original label region should be preserved
        assert np.all(lbl_data[:80, :60, :100] == voxel_labels)
        # Padded region should be 0
        assert np.all(lbl_data[80:, :, :] == 0)

    def test_oversized_sample_clipped(self, tmp_workspace):
        """BDMAP_00002911-like case: Y=129 should be clipped to 128."""
        from convert_hyperbody_to_nnunet import convert_single_sample
        src_dir, out_dir, tmp_path = tmp_workspace

        npz_path = str(src_dir / "BDMAP_00002911.npz")
        sensor_pc, voxel_labels, grid_world_min = make_oversized_npz(npz_path)

        img_dir = out_dir / "imagesTr"
        lbl_dir = out_dir / "labelsTr"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        convert_single_sample(npz_path, str(img_dir), str(lbl_dir))

        lbl_nii = nib.load(str(lbl_dir / "BDMAP_00002911.nii.gz"))
        lbl_data = lbl_nii.get_fdata().astype(np.uint8)
        assert lbl_data.shape == TARGET_SHAPE

        # Verify Y dimension is clipped: only first 128 of 129 rows kept
        assert np.all(lbl_data[:100, :128, :200] == voxel_labels[:100, :128, :200])


class TestDatasetJson:
    """Test dataset.json generation."""

    def test_generate_dataset_json(self, tmp_workspace):
        from convert_hyperbody_to_nnunet import write_dataset_json
        _, out_dir, _ = tmp_workspace
        out_dir.mkdir(parents=True)

        write_dataset_json(str(out_dir), num_training=9779)

        json_path = out_dir / "dataset.json"
        assert json_path.exists()

        with open(json_path) as f:
            ds = json.load(f)

        assert ds["channel_names"] == {"0": "noNorm"}
        assert ds["numTraining"] == 9779
        assert ds["file_ending"] == ".nii.gz"
        assert ds["labels"]["background"] == 0
        assert ds["labels"]["iliopsoas_right"] == 69
        assert len(ds["labels"]) == 70

    def test_label_values_consecutive(self, tmp_workspace):
        """nnUNet requires consecutive label values 0..69."""
        from convert_hyperbody_to_nnunet import write_dataset_json
        _, out_dir, _ = tmp_workspace
        out_dir.mkdir(parents=True)

        write_dataset_json(str(out_dir), num_training=9779)

        with open(out_dir / "dataset.json") as f:
            ds = json.load(f)

        label_values = sorted(ds["labels"].values())
        assert label_values == list(range(70))


class TestFileNaming:
    """Test nnUNet file naming conventions."""

    def test_image_name(self):
        from convert_hyperbody_to_nnunet import get_nnunet_names
        img_name, lbl_name = get_nnunet_names("BDMAP_00004005.npz")
        assert img_name == "BDMAP_00004005_0000.nii.gz"
        assert lbl_name == "BDMAP_00004005.nii.gz"

    def test_case_id_extraction(self):
        from convert_hyperbody_to_nnunet import get_nnunet_names
        img_name, lbl_name = get_nnunet_names("BDMAP_00010934.npz")
        assert img_name == "BDMAP_00010934_0000.nii.gz"
        assert lbl_name == "BDMAP_00010934.nii.gz"


class TestConvertTestLabel:
    """Test convert_test_label: saves test ground-truth labels to NIfTI."""

    def test_produces_label_file(self, tmp_workspace):
        """convert_test_label should create a NIfTI label file in the output dir."""
        from convert_hyperbody_to_nnunet import convert_test_label
        src_dir, out_dir, tmp_path = tmp_workspace

        npz_path = str(src_dir / "BDMAP_00005555.npz")
        make_synthetic_npz(npz_path, grid_shape=(90, 80, 180), seed=77)

        lbl_dir = out_dir / "labelsTs"
        lbl_dir.mkdir(parents=True)

        convert_test_label(npz_path, str(lbl_dir))

        lbl_path = lbl_dir / "BDMAP_00005555.nii.gz"
        assert lbl_path.exists(), f"Label file not found: {lbl_path}"

    def test_output_shape_matches_target(self, tmp_workspace):
        """Output label NIfTI must have shape TARGET_SHAPE."""
        from convert_hyperbody_to_nnunet import convert_test_label
        src_dir, out_dir, tmp_path = tmp_workspace

        npz_path = str(src_dir / "BDMAP_00005556.npz")
        make_synthetic_npz(npz_path, grid_shape=(100, 90, 200), seed=78)

        lbl_dir = out_dir / "labelsTs"
        lbl_dir.mkdir(parents=True)

        convert_test_label(npz_path, str(lbl_dir))

        lbl_nii = nib.load(str(lbl_dir / "BDMAP_00005556.nii.gz"))
        assert lbl_nii.shape == TARGET_SHAPE, (
            f"Expected {TARGET_SHAPE}, got {lbl_nii.shape}"
        )

    def test_label_values_preserved(self, tmp_workspace):
        """Voxel label values must be preserved in the padded NIfTI output."""
        from convert_hyperbody_to_nnunet import convert_test_label
        src_dir, out_dir, tmp_path = tmp_workspace

        npz_path = str(src_dir / "BDMAP_00005557.npz")
        _, voxel_labels, grid_world_min = make_synthetic_npz(
            npz_path, grid_shape=(80, 60, 100), seed=79)

        lbl_dir = out_dir / "labelsTs"
        lbl_dir.mkdir(parents=True)

        convert_test_label(npz_path, str(lbl_dir))

        lbl_nii = nib.load(str(lbl_dir / "BDMAP_00005557.nii.gz"))
        lbl_data = lbl_nii.get_fdata().astype(np.uint8)

        # Original region preserved exactly
        assert np.all(lbl_data[:80, :60, :100] == voxel_labels), (
            "Label values in original region differ from source"
        )
        # Padded region is zeros
        assert np.all(lbl_data[80:, :, :] == 0)
        assert np.all(lbl_data[:, 60:, :] == 0)
        assert np.all(lbl_data[:, :, 100:] == 0)

    def test_file_naming_convention(self, tmp_workspace):
        """Output file must follow BDMAP_XXXXXXXX.nii.gz naming (no _0000 suffix)."""
        from convert_hyperbody_to_nnunet import convert_test_label
        src_dir, out_dir, tmp_path = tmp_workspace

        npz_path = str(src_dir / "BDMAP_00012345.npz")
        make_synthetic_npz(npz_path, grid_shape=(90, 80, 180), seed=80)

        lbl_dir = out_dir / "labelsTs"
        lbl_dir.mkdir(parents=True)

        convert_test_label(npz_path, str(lbl_dir))

        # Correct name exists
        assert (lbl_dir / "BDMAP_00012345.nii.gz").exists()
        # Wrong name (image convention) must NOT exist
        assert not (lbl_dir / "BDMAP_00012345_0000.nii.gz").exists()

    def test_affine_correct(self, tmp_workspace):
        """Affine must reflect voxel spacing and grid origin."""
        from convert_hyperbody_to_nnunet import convert_test_label
        src_dir, out_dir, tmp_path = tmp_workspace

        grid_world_min = np.array([-200.0, -150.0, 50.0], dtype=np.float32)
        npz_path = str(src_dir / "BDMAP_00005558.npz")
        make_synthetic_npz(npz_path, grid_shape=(90, 80, 180),
                           grid_world_min=grid_world_min, seed=81)

        lbl_dir = out_dir / "labelsTs"
        lbl_dir.mkdir(parents=True)

        convert_test_label(npz_path, str(lbl_dir))

        lbl_nii = nib.load(str(lbl_dir / "BDMAP_00005558.nii.gz"))
        affine = lbl_nii.affine
        np.testing.assert_array_almost_equal(
            np.diag(affine), [4.0, 4.0, 4.0, 1.0])
        np.testing.assert_array_almost_equal(
            affine[:3, 3], grid_world_min)

    def test_oversized_labels_clipped(self, tmp_workspace):
        """Labels exceeding TARGET_SHAPE should be clipped, not error."""
        from convert_hyperbody_to_nnunet import convert_test_label
        src_dir, out_dir, tmp_path = tmp_workspace

        npz_path = str(src_dir / "BDMAP_00005559.npz")
        make_synthetic_npz(npz_path, grid_shape=(100, 129, 200), seed=82)

        lbl_dir = out_dir / "labelsTs"
        lbl_dir.mkdir(parents=True)

        convert_test_label(npz_path, str(lbl_dir))

        lbl_nii = nib.load(str(lbl_dir / "BDMAP_00005559.nii.gz"))
        assert lbl_nii.shape == TARGET_SHAPE

    def test_dtype_uint8(self, tmp_workspace):
        """Saved label data must be uint8 to match nnUNet expectations."""
        from convert_hyperbody_to_nnunet import convert_test_label
        src_dir, out_dir, tmp_path = tmp_workspace

        npz_path = str(src_dir / "BDMAP_00005560.npz")
        make_synthetic_npz(npz_path, grid_shape=(90, 80, 180), seed=83)

        lbl_dir = out_dir / "labelsTs"
        lbl_dir.mkdir(parents=True)

        convert_test_label(npz_path, str(lbl_dir))

        lbl_nii = nib.load(str(lbl_dir / "BDMAP_00005560.nii.gz"))
        assert lbl_nii.get_data_dtype() == np.uint8, (
            f"Expected uint8, got {lbl_nii.get_data_dtype()}"
        )


class TestRealTestLabelSanity:
    """Integration test on a real test sample (if available)."""

    REAL_DATA_DIR = "/home/comp/csrkzhu/code/Compare/nnUNet/Dataset/voxel_data"
    SPLIT_JSON = "/home/comp/csrkzhu/code/Compare/nnUNet/Dataset/dataset_split.json"

    @pytest.mark.skipif(
        not os.path.exists(
            "/home/comp/csrkzhu/code/Compare/nnUNet/Dataset/dataset_split.json"),
        reason="Split JSON not available"
    )
    def test_real_test_sample_converts(self, tmp_path):
        from convert_hyperbody_to_nnunet import convert_test_label

        with open(self.SPLIT_JSON) as f:
            split = json.load(f)
        test_file = split["test"][0]
        npz_path = os.path.join(self.REAL_DATA_DIR, test_file)

        if not os.path.exists(npz_path):
            pytest.skip(f"Real test data not available: {npz_path}")

        lbl_dir = str(tmp_path / "labelsTs")
        os.makedirs(lbl_dir)

        convert_test_label(npz_path, lbl_dir)

        case_id = test_file.replace(".npz", "")
        lbl_path = os.path.join(lbl_dir, f"{case_id}.nii.gz")
        assert os.path.exists(lbl_path)

        lbl_nii = nib.load(lbl_path)
        assert lbl_nii.shape == TARGET_SHAPE
        lbl_data = lbl_nii.get_fdata().astype(np.uint8)
        assert lbl_data.max() < NUM_CLASSES
        assert lbl_nii.get_data_dtype() == np.uint8

        # Print summary for visual inspection
        print(f"\n--- Real test label: {test_file} ---")
        print(f"  Shape: {lbl_nii.shape}")
        print(f"  Unique labels: {np.unique(lbl_data)}")
        print(f"  Non-zero voxels: {np.count_nonzero(lbl_data)}")
        print(f"  Affine origin: {lbl_nii.affine[:3, 3]}")


class TestRealDataSanity:
    """Sanity check on a real .npz file (if available)."""

    REAL_DATA_DIR = "/home/comp/csrkzhu/code/Compare/nnUNet/Dataset/voxel_data"
    SAMPLE_FILE = "BDMAP_00000001.npz"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(REAL_DATA_DIR, SAMPLE_FILE)),
        reason="Real data not available"
    )
    def test_real_sample_loads_and_converts(self, tmp_path):
        from convert_hyperbody_to_nnunet import convert_single_sample

        npz_path = os.path.join(self.REAL_DATA_DIR, self.SAMPLE_FILE)
        img_dir = str(tmp_path / "imagesTr")
        lbl_dir = str(tmp_path / "labelsTr")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        convert_single_sample(npz_path, img_dir, lbl_dir)

        # Verify output files
        case_id = self.SAMPLE_FILE.replace(".npz", "")
        img_path = os.path.join(img_dir, f"{case_id}_0000.nii.gz")
        lbl_path = os.path.join(lbl_dir, f"{case_id}.nii.gz")
        assert os.path.exists(img_path)
        assert os.path.exists(lbl_path)

        # Load and check shapes
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        assert img_nii.shape == TARGET_SHAPE
        assert lbl_nii.shape == TARGET_SHAPE

        # Image should be binary
        img_data = img_nii.get_fdata()
        assert set(np.unique(img_data)).issubset({0.0, 1.0})

        # Label should have valid class range
        lbl_data = lbl_nii.get_fdata().astype(np.uint8)
        assert lbl_data.max() < NUM_CLASSES

        # Affine should have 4mm spacing
        np.testing.assert_array_almost_equal(
            np.diag(img_nii.affine)[:3], [4.0, 4.0, 4.0])

        # Print summary for visual inspection
        print(f"\n--- Real sample test: {self.SAMPLE_FILE} ---")
        print(f"  Image shape: {img_nii.shape}")
        print(f"  Label shape: {lbl_nii.shape}")
        print(f"  Occupancy density: {img_data.sum() / img_data.size * 100:.2f}%")
        print(f"  Unique labels: {np.unique(lbl_data)}")
        print(f"  Affine origin: {img_nii.affine[:3, 3]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
