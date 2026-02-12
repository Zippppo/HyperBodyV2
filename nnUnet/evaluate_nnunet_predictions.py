"""
Evaluate nnUNet predictions against ground truth for the HyperBody dataset.

Computes per-class Dice scores, mean Dice (excluding background), and
per-organ-group Dice. Supports evaluation in both padded space (144,128,268)
and original space (cropped to grid_occ_size from the .npz files).

Usage:
    # Padded space evaluation
    python evaluate_nnunet_predictions.py \
        --pred_dir nnUNet_data/nnUNet_results/Dataset501_HyperBody/predictions/ \
        --gt_dir nnUNet_data/nnUNet_raw/Dataset501_HyperBody/labelsTs/ \
        --output_dir evaluation_results/

    # Original space evaluation (requires .npz files)
    python evaluate_nnunet_predictions.py \
        --pred_dir nnUNet_data/nnUNet_results/Dataset501_HyperBody/predictions/ \
        --gt_dir nnUNet_data/nnUNet_raw/Dataset501_HyperBody/labelsTs/ \
        --output_dir evaluation_results/ \
        --npz_dir Dataset/voxel_data/ \
        --original_space
"""
import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- Constants ----
TARGET_SHAPE = (144, 128, 268)
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

# Reverse lookup: class_id -> name
LABEL_NAMES = {v: k for k, v in LABELS.items()}

# Organ groups for grouped metrics
ORGAN_GROUPS = {
    "organs": list(range(1, 13)),        # classes 1-12
    "soft_tissue": list(range(13, 22)),   # classes 13-21
    "bones": list(range(22, 60)),         # classes 22-59
    "muscles": list(range(60, 70)),       # classes 60-69
}


# ============================================================
# Core evaluation functions
# ============================================================

def compute_dice_per_class(pred, gt, num_classes=70):
    """
    Compute per-class Dice coefficient between prediction and ground truth.

    Uses vectorized numpy operations: for each class, creates boolean masks
    and computes Dice = 2 * |intersection| / (|pred| + |gt|).

    Args:
        pred: numpy array (D, H, W) of integer class labels
        gt: numpy array (D, H, W) of integer class labels
        num_classes: number of classes (0 to num_classes-1)

    Returns:
        dict of {class_id: dice_score}, where dice_score is float or NaN
        (NaN if the class is absent in both pred and gt)
    """
    pred = pred.ravel()
    gt = gt.ravel()

    dice_dict = {}
    for c in range(num_classes):
        pred_mask = (pred == c)
        gt_mask = (gt == c)

        pred_sum = pred_mask.sum()
        gt_sum = gt_mask.sum()

        if pred_sum == 0 and gt_sum == 0:
            # Class absent in both -> undefined
            dice_dict[c] = float('nan')
        else:
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            dice_dict[c] = 2.0 * intersection / (pred_sum + gt_sum)

    return dice_dict


def compute_mean_dice(dice_dict, exclude_background=True):
    """
    Compute mean Dice score from a per-class Dice dictionary.

    Args:
        dice_dict: dict of {class_id: dice_score}
        exclude_background: if True, skip class 0

    Returns:
        float: mean Dice (NaN if no valid scores)
    """
    values = []
    for c, d in dice_dict.items():
        if exclude_background and c == 0:
            continue
        if not np.isnan(d):
            values.append(d)

    if not values:
        return float('nan')
    return float(np.mean(values))


def compute_group_dice(dice_dict):
    """
    Compute mean Dice per organ group.

    Args:
        dice_dict: dict of {class_id: dice_score}

    Returns:
        dict of {group_name: mean_dice}
    """
    group_dice = {}
    for group_name, class_ids in ORGAN_GROUPS.items():
        values = []
        for c in class_ids:
            if c in dice_dict and not np.isnan(dice_dict[c]):
                values.append(dice_dict[c])
        if values:
            group_dice[group_name] = float(np.mean(values))
        else:
            group_dice[group_name] = float('nan')
    return group_dice


# ============================================================
# Sample-level evaluation
# ============================================================

def evaluate_single_sample(pred_path, gt_path, num_classes=70):
    """
    Evaluate a single sample: load NIfTI files and compute per-class Dice.

    Args:
        pred_path: path to prediction NIfTI file
        gt_path: path to ground truth NIfTI file
        num_classes: number of classes

    Returns:
        dict of {class_id: dice_score}
    """
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)

    pred_data = np.asarray(pred_nii.dataobj, dtype=np.uint8)
    gt_data = np.asarray(gt_nii.dataobj, dtype=np.uint8)

    return compute_dice_per_class(pred_data, gt_data, num_classes=num_classes)


def evaluate_single_sample_original_space(pred_path, gt_path, npz_path,
                                           num_classes=70):
    """
    Evaluate a single sample in original (unpadded) space.

    Crops both prediction and ground truth to the original grid_occ_size
    read from the .npz file before computing Dice.

    Args:
        pred_path: path to prediction NIfTI file
        gt_path: path to ground truth NIfTI file
        npz_path: path to original .npz file (for grid_occ_size)
        num_classes: number of classes

    Returns:
        dict of {class_id: dice_score}
    """
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)

    pred_data = np.asarray(pred_nii.dataobj, dtype=np.uint8)
    gt_data = np.asarray(gt_nii.dataobj, dtype=np.uint8)

    # Load original grid size
    npz_data = np.load(npz_path)
    grid_occ_size = npz_data["grid_occ_size"]  # (3,) int32

    # Crop to original space (corner-aligned, same as pad_labels)
    d, h, w = int(grid_occ_size[0]), int(grid_occ_size[1]), int(grid_occ_size[2])
    pred_cropped = pred_data[:d, :h, :w]
    gt_cropped = gt_data[:d, :h, :w]

    return compute_dice_per_class(pred_cropped, gt_cropped, num_classes=num_classes)


# ============================================================
# Folder-level evaluation
# ============================================================

def _evaluate_worker(args):
    """Worker function for parallel evaluation."""
    case_id, pred_path, gt_path, num_classes, npz_path = args
    try:
        if npz_path is not None:
            dice_dict = evaluate_single_sample_original_space(
                pred_path, gt_path, npz_path, num_classes=num_classes)
        else:
            dice_dict = evaluate_single_sample(
                pred_path, gt_path, num_classes=num_classes)
        return case_id, dice_dict, ""
    except Exception as e:
        return case_id, None, str(e)


def evaluate_folder(pred_dir, gt_dir, num_classes=70, num_workers=8,
                    npz_dir=None):
    """
    Evaluate all matching prediction/ground-truth pairs in a directory.

    Args:
        pred_dir: directory with prediction NIfTI files
        gt_dir: directory with ground truth NIfTI files
        num_classes: number of classes
        num_workers: number of parallel workers
        npz_dir: optional, directory with .npz files for original-space eval

    Returns:
        dict of {case_id: dice_dict}
    """
    # Find matching files
    pred_files = {f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")}
    gt_files = {f for f in os.listdir(gt_dir) if f.endswith(".nii.gz")}
    common_files = sorted(pred_files & gt_files)

    if not common_files:
        print("WARNING: No matching files found between pred_dir and gt_dir")
        return {}

    print(f"Found {len(common_files)} matching files to evaluate")

    # Prepare tasks
    tasks = []
    for fname in common_files:
        case_id = fname.replace(".nii.gz", "")
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        npz_path = None
        if npz_dir is not None:
            npz_path = os.path.join(npz_dir, f"{case_id}.npz")
            if not os.path.exists(npz_path):
                print(f"WARNING: NPZ file not found for {case_id}, "
                      f"skipping original-space eval")
                npz_path = None

        tasks.append((case_id, pred_path, gt_path, num_classes, npz_path))

    # Execute in parallel
    results = {}
    failed = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_evaluate_worker, t): t[0] for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            case_id, dice_dict, err = future.result()
            if dice_dict is not None:
                results[case_id] = dice_dict
            else:
                failed.append((case_id, err))

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Progress: {i + 1}/{len(tasks)}")

    if failed:
        print(f"\nWARNING: {len(failed)} evaluations failed:")
        for case_id, err in failed[:10]:
            print(f"  {case_id}: {err}")

    return results


# ============================================================
# Aggregation helper
# ============================================================

def aggregate_results(results, num_classes=70):
    """
    Compute per-class mean, overall mean Dice, and group Dice from per-case results.

    Args:
        results: dict of {case_id: {class_id: dice_score}}
        num_classes: number of classes

    Returns:
        dict with keys:
            "per_class_mean": {class_id(int): float or NaN}
            "mean_dice": float or NaN
            "group_dice": {group_name: float or NaN}
    """
    all_classes = set()
    for dice_dict in results.values():
        all_classes.update(dice_dict.keys())

    # Per-class mean across all cases
    per_class_mean = {}
    for c in sorted(all_classes):
        values = []
        for dice_dict in results.values():
            if c in dice_dict and not np.isnan(dice_dict[c]):
                values.append(dice_dict[c])
        per_class_mean[c] = float(np.mean(values)) if values else float('nan')

    # Overall mean Dice (excluding background, across all cases)
    case_means = []
    for dice_dict in results.values():
        m = compute_mean_dice(dice_dict, exclude_background=True)
        if not np.isnan(m):
            case_means.append(m)
    mean_dice = float(np.mean(case_means)) if case_means else float('nan')

    # Group Dice from per_class_mean
    group_dice = compute_group_dice(per_class_mean)

    return {
        "per_class_mean": per_class_mean,
        "mean_dice": mean_dice,
        "group_dice": group_dice,
    }


# ============================================================
# Results output
# ============================================================

def save_results(results, output_path):
    """
    Save evaluation results to a JSON file.

    Output JSON structure:
    {
        "per_case": {case_id: {class_id: dice, ...}, ...},
        "mean_dice": float,
        "per_class_mean": {class_id: mean_dice_across_cases, ...},
        "group_dice": {group_name: mean_dice, ...}
    }

    Args:
        results: dict of {case_id: {class_id: dice_score}}
        output_path: path to save JSON file
    """
    # Convert per-case dice dicts: class_id (int) -> str for JSON, NaN -> None
    per_case = {}
    for case_id, dice_dict in results.items():
        per_case[case_id] = {
            str(c): (None if np.isnan(d) else float(d))
            for c, d in dice_dict.items()
        }

    # Use shared aggregation helper
    agg = aggregate_results(results)
    mean_dice = agg["mean_dice"]
    per_class_mean_raw = agg["per_class_mean"]
    group_dice_raw = agg["group_dice"]

    # Convert to JSON-safe format: NaN -> None, int keys -> str keys
    per_class_mean = {
        str(c): (None if np.isnan(v) else float(v))
        for c, v in per_class_mean_raw.items()
    }
    group_dice = {
        k: (None if np.isnan(v) else float(v))
        for k, v in group_dice_raw.items()
    }

    output = {
        "mean_dice": None if np.isnan(mean_dice) else float(mean_dice),
        "group_dice": group_dice,
        "per_class_mean": per_class_mean,
        "per_case": per_case,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"Results saved to {output_path}")


def _json_default(obj):
    """Handle non-serializable objects in JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def plot_per_class_dice(mean_dice_dict, output_path):
    """
    Generate a bar chart of per-class mean Dice scores and save as PNG.

    Args:
        mean_dice_dict: dict of {class_id: mean_dice} (int keys)
        output_path: path to save PNG file
    """
    # Sort by class ID
    class_ids = sorted(mean_dice_dict.keys())
    dice_values = []
    class_names = []

    for c in class_ids:
        val = mean_dice_dict[c]
        dice_values.append(0.0 if np.isnan(val) else val)
        class_names.append(LABEL_NAMES.get(c, f"class_{c}"))

    # Assign colors by organ group
    colors = []
    group_colors = {
        "organs": "#4CAF50",       # green
        "soft_tissue": "#2196F3",  # blue
        "bones": "#FF9800",        # orange
        "muscles": "#E91E63",      # pink
    }
    for c in class_ids:
        color = "#9E9E9E"  # grey for background/unassigned
        for group_name, group_ids in ORGAN_GROUPS.items():
            if c in group_ids:
                color = group_colors[group_name]
                break
        colors.append(color)

    fig, ax = plt.subplots(figsize=(24, 8))
    x = np.arange(len(class_ids))
    ax.bar(x, dice_values, color=colors, width=0.8, edgecolor="white",
           linewidth=0.3)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Per-Class Mean Dice Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6, ha="center")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.nanmean(dice_values), color="red", linestyle="--",
               linewidth=1, label=f"Mean: {np.nanmean(dice_values):.3f}")

    # Add legend for groups
    legend_elements = [
        Patch(facecolor=group_colors["organs"], label="Organs (1-12)"),
        Patch(facecolor=group_colors["soft_tissue"], label="Soft Tissue (13-21)"),
        Patch(facecolor=group_colors["bones"], label="Bones (22-59)"),
        Patch(facecolor=group_colors["muscles"], label="Muscles (60-69)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ============================================================
# CLI main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate nnUNet predictions for HyperBody dataset")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction NIfTI files")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth NIfTI files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--npz_dir", type=str, default=None,
                        help="Directory containing .npz files (for original-space eval)")
    parser.add_argument("--original_space", action="store_true",
                        help="Evaluate in original (unpadded) space using grid_occ_size")
    parser.add_argument("--num_classes", type=int, default=70,
                        help="Number of classes (default: 70)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    # Validate
    if args.original_space and args.npz_dir is None:
        parser.error("--npz_dir is required when --original_space is set")

    print(f"Prediction directory: {args.pred_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.original_space:
        print(f"NPZ directory: {args.npz_dir}")
        print("Evaluation mode: original space (cropped to grid_occ_size)")
    else:
        print("Evaluation mode: padded space")

    # Run evaluation
    npz_dir = args.npz_dir if args.original_space else None
    results = evaluate_folder(
        args.pred_dir, args.gt_dir,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
        npz_dir=npz_dir,
    )

    if not results:
        print("No results to save. Exiting.")
        return

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    mode_suffix = "original_space" if args.original_space else "padded_space"
    json_path = os.path.join(args.output_dir, f"results_{mode_suffix}.json")
    save_results(results, json_path)

    # Use shared aggregation helper
    agg = aggregate_results(results, num_classes=args.num_classes)
    per_class_mean = agg["per_class_mean"]
    overall_mean = agg["mean_dice"]
    group_dice = agg["group_dice"]

    # Plot
    plot_path = os.path.join(args.output_dir, f"dice_per_class_{mode_suffix}.png")
    plot_per_class_dice(per_class_mean, plot_path)

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({mode_suffix})")
    print(f"{'='*60}")
    print(f"Total cases evaluated: {len(results)}")
    print(f"Overall mean Dice (excl. background): {overall_mean:.4f}")
    print(f"\nPer-group Dice:")
    for group_name, gd in group_dice.items():
        if np.isnan(gd):
            print(f"  {group_name:15s}: N/A")
        else:
            print(f"  {group_name:15s}: {gd:.4f}")

    print(f"\nTop 10 classes by Dice:")
    sorted_classes = sorted(
        [(c, d) for c, d in per_class_mean.items()
         if c != 0 and not np.isnan(d)],
        key=lambda x: x[1], reverse=True
    )
    for c, d in sorted_classes[:10]:
        print(f"  {LABEL_NAMES.get(c, f'class_{c}'):30s} (class {c:2d}): {d:.4f}")

    print(f"\nBottom 10 classes by Dice:")
    for c, d in sorted_classes[-10:]:
        print(f"  {LABEL_NAMES.get(c, f'class_{c}'):30s} (class {c:2d}): {d:.4f}")

    print(f"\nResults saved to: {json_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
