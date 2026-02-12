"""
Convert HyperBody dataset (.npz) to nnUNet format (NIfTI).

Input:  Dataset/voxel_data/*.npz  (sensor_pc + voxel_labels)
Output: nnUNet_data/nnUNet_raw/Dataset501_HyperBody/
        ├── dataset.json
        ├── imagesTr/   BDMAP_XXXXXXXX_0000.nii.gz  (binary occupancy grid)
        ├── labelsTr/   BDMAP_XXXXXXXX.nii.gz       (voxel labels)
        ├── imagesTs/   BDMAP_XXXXXXXX_0000.nii.gz  (test occupancy grid)
        └── labelsTs/   BDMAP_XXXXXXXX.nii.gz       (test ground truth labels)

Usage:
    python convert_hyperbody_to_nnunet.py                       # full pipeline
    python convert_hyperbody_to_nnunet.py --create_test_labels  # test labels only
"""
import os
import json
import argparse
from multiprocessing import Pool
from functools import partial

import numpy as np
import nibabel as nib

# ---- Constants ----
TARGET_SHAPE = (144, 128, 268)
VOXEL_SIZE = np.array([4.0, 4.0, 4.0], dtype=np.float32)
NUM_CLASSES = 70

LABELS = {
    "background": 0, "liver": 1, "spleen": 2, "kidney_left": 3,
    "kidney_right": 4, "stomach": 5, "pancreas": 6, "gallbladder": 7,
    "urinary_bladder": 8, "prostate": 9, "heart": 10, "brain": 11,
    "thyroid_gland": 12, "spinal_cord": 13, "lung": 14, "esophagus": 15,
    "trachea": 16, "small_bowel": 17, "duodenum": 18, "colon": 19,
    "adrenal_gland_left": 20, "adrenal_gland_right": 21, "spine": 22,
    "rib_left_1": 23, "rib_left_2": 24, "rib_left_3": 25, "rib_left_4": 26,
    "rib_left_5": 27, "rib_left_6": 28, "rib_left_7": 29, "rib_left_8": 30,
    "rib_left_9": 31, "rib_left_10": 32, "rib_left_11": 33, "rib_left_12": 34,
    "rib_right_1": 35, "rib_right_2": 36, "rib_right_3": 37, "rib_right_4": 38,
    "rib_right_5": 39, "rib_right_6": 40, "rib_right_7": 41, "rib_right_8": 42,
    "rib_right_9": 43, "rib_right_10": 44, "rib_right_11": 45, "rib_right_12": 46,
    "skull": 47, "sternum": 48, "costal_cartilages": 49,
    "scapula_left": 50, "scapula_right": 51,
    "clavicula_left": 52, "clavicula_right": 53,
    "humerus_left": 54, "humerus_right": 55,
    "hip_left": 56, "hip_right": 57,
    "femur_left": 58, "femur_right": 59,
    "gluteus_maximus_left": 60, "gluteus_maximus_right": 61,
    "gluteus_medius_left": 62, "gluteus_medius_right": 63,
    "gluteus_minimus_left": 64, "gluteus_minimus_right": 65,
    "autochthon_left": 66, "autochthon_right": 67,
    "iliopsoas_left": 68, "iliopsoas_right": 69,
}


# ============================================================
# Core conversion functions
# ============================================================

def build_occupancy_grid(sensor_pc, grid_world_min, grid_voxel_size, target_shape):
    """
    Convert point cloud to binary occupancy grid.

    Args:
        sensor_pc: (N, 3) float32 point cloud in world coordinates
        grid_world_min: (3,) float32 world coordinate origin
        grid_voxel_size: (3,) float32 voxel size in mm
        target_shape: (3,) target grid dimensions

    Returns:
        occupancy: float32 array of target_shape, values 0.0 or 1.0
    """
    idx = np.floor((sensor_pc - grid_world_min) / grid_voxel_size).astype(np.int64)
    for d in range(3):
        idx[:, d] = np.clip(idx[:, d], 0, target_shape[d] - 1)

    occupancy = np.zeros(target_shape, dtype=np.float32)
    occupancy[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return occupancy


def pad_labels(voxel_labels, target_shape):
    """
    Corner-aligned pad (or clip) labels to target shape.

    Args:
        voxel_labels: (X, Y, Z) uint8 ground truth
        target_shape: (3,) target dimensions

    Returns:
        label: uint8 array of target_shape
    """
    label = np.zeros(target_shape, dtype=np.uint8)
    x, y, z = voxel_labels.shape
    cx = min(x, target_shape[0])
    cy = min(y, target_shape[1])
    cz = min(z, target_shape[2])
    label[:cx, :cy, :cz] = voxel_labels[:cx, :cy, :cz]
    return label


def create_nifti_affine(grid_world_min, voxel_size):
    """
    Create NIfTI affine matrix with correct spacing and origin.

    Args:
        grid_world_min: (3,) float32 world coordinate origin
        voxel_size: (3,) float32 voxel spacing in mm

    Returns:
        affine: (4, 4) float64 affine matrix
    """
    affine = np.diag([float(voxel_size[0]), float(voxel_size[1]),
                      float(voxel_size[2]), 1.0])
    affine[:3, 3] = grid_world_min
    return affine


def get_nnunet_names(npz_filename):
    """
    Get nnUNet naming convention for a given .npz filename.

    Args:
        npz_filename: e.g. "BDMAP_00004005.npz"

    Returns:
        (image_name, label_name): e.g. ("BDMAP_00004005_0000.nii.gz", "BDMAP_00004005.nii.gz")
    """
    case_id = npz_filename.replace(".npz", "")
    return f"{case_id}_0000.nii.gz", f"{case_id}.nii.gz"


def convert_single_sample(npz_path, img_out_dir, lbl_out_dir):
    """
    Convert a single .npz sample to nnUNet NIfTI format.

    Args:
        npz_path: path to input .npz file
        img_out_dir: directory for output image NIfTI
        lbl_out_dir: directory for output label NIfTI
    """
    data = np.load(npz_path)
    sensor_pc = data["sensor_pc"]
    voxel_labels = data["voxel_labels"]
    grid_world_min = data["grid_world_min"]
    grid_voxel_size = data["grid_voxel_size"]

    # Build occupancy grid
    occupancy = build_occupancy_grid(sensor_pc, grid_world_min, grid_voxel_size, TARGET_SHAPE)

    # Pad/clip labels
    label = pad_labels(voxel_labels, TARGET_SHAPE)

    # Create affine
    affine = create_nifti_affine(grid_world_min, grid_voxel_size)

    # Get file names
    npz_filename = os.path.basename(npz_path)
    img_name, lbl_name = get_nnunet_names(npz_filename)

    # Save NIfTI
    img_nii = nib.Nifti1Image(occupancy, affine)
    nib.save(img_nii, os.path.join(img_out_dir, img_name))

    lbl_nii = nib.Nifti1Image(label, affine)
    nib.save(lbl_nii, os.path.join(lbl_out_dir, lbl_name))


def write_dataset_json(output_dir, num_training=9779):
    """
    Write dataset.json for nnUNet.

    Args:
        output_dir: Dataset501_HyperBody directory
        num_training: number of training cases
    """
    dataset_json = {
        "channel_names": {"0": "noNorm"},
        "labels": LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)


# ============================================================
# Worker for multiprocessing
# ============================================================

def _convert_worker(args):
    """Worker function for parallel conversion."""
    npz_path, img_out_dir, lbl_out_dir = args
    try:
        convert_single_sample(npz_path, img_out_dir, lbl_out_dir)
        return npz_path, True, ""
    except Exception as e:
        return npz_path, False, str(e)


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Convert HyperBody to nnUNet format")
    parser.add_argument("--data_dir", type=str,
                        default="/home/comp/csrkzhu/code/Compare/nnUNet/Dataset/voxel_data",
                        help="Directory containing .npz files")
    parser.add_argument("--split_json", type=str,
                        default="/home/comp/csrkzhu/code/Compare/nnUNet/Dataset/dataset_split.json",
                        help="Path to dataset_split.json")
    parser.add_argument("--output_dir", type=str,
                        default="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_raw/Dataset501_HyperBody",
                        help="Output directory for nnUNet dataset")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--create_test_labels", action="store_true",
                        help="Create ground truth labels for test set (labelsTs/)")
    parser.add_argument("--convert_val_to_train", action="store_true",
                        help="Convert val split samples into imagesTr/labelsTr (append to training set)")
    args = parser.parse_args()

    # Load split
    with open(args.split_json) as f:
        split = json.load(f)

    train_files = split["train"]
    test_files = split["test"]

    print(f"Train samples: {len(train_files)}")
    print(f"Test samples:  {len(test_files)}")

    # --create_test_labels mode: only generate labelsTs/ and exit
    if args.create_test_labels:
        labels_ts = os.path.join(args.output_dir, "labelsTs")
        os.makedirs(labels_ts, exist_ok=True)

        print(f"\nCreating {len(test_files)} test labels in {labels_ts}...")
        test_label_tasks = [
            (os.path.join(args.data_dir, f), labels_ts)
            for f in test_files
        ]
        failed = []
        with Pool(args.num_workers) as pool:
            for i, (path, success, err) in enumerate(
                    pool.imap_unordered(_convert_test_label_worker, test_label_tasks)):
                if (i + 1) % 100 == 0 or i == 0:
                    print(f"  Progress: {i + 1}/{len(test_label_tasks)}")
                if not success:
                    failed.append((path, err))

        if failed:
            print(f"\nWARNING: {len(failed)} test label conversions failed:")
            for path, err in failed[:10]:
                print(f"  {path}: {err}")

        print(f"\nTest label creation complete!")
        print(f"  Test labels: {len(os.listdir(labels_ts))}")
        return

    # --convert_val_to_train mode: convert val samples into imagesTr/labelsTr and exit
    if args.convert_val_to_train:
        val_files = split["val"]
        images_tr = os.path.join(args.output_dir, "imagesTr")
        labels_tr = os.path.join(args.output_dir, "labelsTr")
        os.makedirs(images_tr, exist_ok=True)
        os.makedirs(labels_tr, exist_ok=True)

        print(f"\nConverting {len(val_files)} val samples into imagesTr/labelsTr...")
        val_tasks = [
            (os.path.join(args.data_dir, f), images_tr, labels_tr)
            for f in val_files
        ]
        failed = []
        with Pool(args.num_workers) as pool:
            for i, (path, success, err) in enumerate(
                    pool.imap_unordered(_convert_worker, val_tasks)):
                if (i + 1) % 100 == 0 or i == 0:
                    print(f"  Progress: {i + 1}/{len(val_tasks)}")
                if not success:
                    failed.append((path, err))

        if failed:
            print(f"\nWARNING: {len(failed)} val conversions failed:")
            for path, err in failed[:10]:
                print(f"  {path}: {err}")

        print(f"\nVal-to-train conversion complete!")
        print(f"  Total images in imagesTr: {len(os.listdir(images_tr))}")
        print(f"  Total labels in labelsTr: {len(os.listdir(labels_tr))}")
        return

    # Create output directories
    images_tr = os.path.join(args.output_dir, "imagesTr")
    labels_tr = os.path.join(args.output_dir, "labelsTr")
    images_ts = os.path.join(args.output_dir, "imagesTs")
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)

    # Write dataset.json
    write_dataset_json(args.output_dir, num_training=len(train_files))
    print(f"Wrote dataset.json to {args.output_dir}")

    # Prepare conversion tasks
    train_tasks = [
        (os.path.join(args.data_dir, f), images_tr, labels_tr)
        for f in train_files
    ]
    test_tasks = [
        (os.path.join(args.data_dir, f), images_ts, None)
        for f in test_files
    ]

    # Convert training samples
    print(f"\nConverting {len(train_tasks)} training samples...")
    failed = []
    with Pool(args.num_workers) as pool:
        for i, (path, success, err) in enumerate(pool.imap_unordered(_convert_worker, train_tasks)):
            if (i + 1) % 500 == 0 or i == 0:
                print(f"  Progress: {i + 1}/{len(train_tasks)}")
            if not success:
                failed.append((path, err))

    if failed:
        print(f"\nWARNING: {len(failed)} training samples failed:")
        for path, err in failed[:10]:
            print(f"  {path}: {err}")

    # Convert test samples (images only, no labels)
    print(f"\nConverting {len(test_tasks)} test samples...")

    # For test, we only save images (no labels directory)
    test_tasks_img_only = [
        (os.path.join(args.data_dir, f), images_ts, images_ts)
        for f in test_files
    ]
    # Override: test samples should not save labels, need a special worker
    for npz_file in test_files:
        npz_path = os.path.join(args.data_dir, npz_file)
        convert_test_sample(npz_path, images_ts)

    print(f"\nConversion complete!")
    print(f"  Train images: {len(os.listdir(images_tr))}")
    print(f"  Train labels: {len(os.listdir(labels_tr))}")
    print(f"  Test images:  {len(os.listdir(images_ts))}")


def convert_test_sample(npz_path, img_out_dir):
    """Convert a test sample (image only, no label)."""
    data = np.load(npz_path)
    sensor_pc = data["sensor_pc"]
    grid_world_min = data["grid_world_min"]
    grid_voxel_size = data["grid_voxel_size"]

    occupancy = build_occupancy_grid(sensor_pc, grid_world_min, grid_voxel_size, TARGET_SHAPE)
    affine = create_nifti_affine(grid_world_min, grid_voxel_size)

    npz_filename = os.path.basename(npz_path)
    img_name, _ = get_nnunet_names(npz_filename)

    img_nii = nib.Nifti1Image(occupancy, affine)
    nib.save(img_nii, os.path.join(img_out_dir, img_name))


def convert_test_label(npz_path, lbl_out_dir):
    """
    Convert a test sample's voxel_labels to NIfTI ground truth label.

    Reuses pad_labels() and create_nifti_affine() for consistency
    with training label conversion.

    Args:
        npz_path: path to input .npz file
        lbl_out_dir: directory for output label NIfTI (labelsTs/)
    """
    data = np.load(npz_path)
    voxel_labels = data["voxel_labels"]
    grid_world_min = data["grid_world_min"]
    grid_voxel_size = data["grid_voxel_size"]

    # Pad/clip labels to target shape (same as training)
    label = pad_labels(voxel_labels, TARGET_SHAPE)

    # Create affine with correct spacing and origin
    affine = create_nifti_affine(grid_world_min, grid_voxel_size)

    # Get label file name (BDMAP_XXXXXXXX.nii.gz)
    npz_filename = os.path.basename(npz_path)
    _, lbl_name = get_nnunet_names(npz_filename)

    # Save NIfTI
    lbl_nii = nib.Nifti1Image(label, affine)
    nib.save(lbl_nii, os.path.join(lbl_out_dir, lbl_name))


def _convert_test_label_worker(args):
    """Worker function for parallel test label conversion."""
    npz_path, lbl_out_dir = args
    try:
        convert_test_label(npz_path, lbl_out_dir)
        return npz_path, True, ""
    except Exception as e:
        return npz_path, False, str(e)


if __name__ == "__main__":
    main()
