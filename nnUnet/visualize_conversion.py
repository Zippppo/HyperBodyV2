"""
Visualization script to verify HyperBody -> nnUNet data conversion.

Produces interactive Plotly HTML visualizations comparing:
1. Original point cloud vs voxelized occupancy grid
2. Original voxel labels vs padded NIfTI labels
3. Multiple samples for sanity check

Usage:
    python visualize_conversion.py
"""
import os
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ---- Paths ----
PROJECT_ROOT = "/home/comp/csrkzhu/code/Compare/nnUNet"
NPZ_DIR = os.path.join(PROJECT_ROOT, "Dataset/voxel_data")
NNUNET_DIR = os.path.join(PROJECT_ROOT, "nnUNet_data/nnUNet_raw/Dataset501_HyperBody")
SPLIT_JSON = os.path.join(PROJECT_ROOT, "Dataset/dataset_split.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "docs/verification")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SHAPE = (144, 128, 268)


def load_original(case_id):
    """Load original .npz data."""
    data = np.load(os.path.join(NPZ_DIR, f"{case_id}.npz"))
    return {
        "sensor_pc": data["sensor_pc"],
        "voxel_labels": data["voxel_labels"],
        "grid_world_min": data["grid_world_min"],
        "grid_voxel_size": data["grid_voxel_size"],
        "grid_occ_size": data["grid_occ_size"],
    }


def load_converted(case_id, split="train"):
    """Load converted NIfTI data."""
    img_subdir = "imagesTr" if split == "train" else "imagesTs"
    img_path = os.path.join(NNUNET_DIR, img_subdir, f"{case_id}_0000.nii.gz")
    img_nii = nib.load(img_path)
    result = {"occupancy": img_nii.get_fdata(), "affine": img_nii.affine}

    if split == "train":
        lbl_path = os.path.join(NNUNET_DIR, "labelsTr", f"{case_id}.nii.gz")
        lbl_nii = nib.load(lbl_path)
        result["labels"] = lbl_nii.get_fdata().astype(np.uint8)
        result["label_affine"] = lbl_nii.affine

    return result


def subsample_points(points, max_points=5000, seed=42):
    """Subsample points for visualization performance."""
    if len(points) <= max_points:
        return points
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(points), max_points, replace=False)
    return points[idx]


def voxel_to_world(voxel_indices, affine):
    """Convert voxel indices to world coordinates using affine."""
    ones = np.ones((len(voxel_indices), 1))
    voxel_homog = np.hstack([voxel_indices, ones])
    world = (affine @ voxel_homog.T).T[:, :3]
    return world


def visualize_sample(case_id, split="train"):
    """Create interactive 3D visualization for a single sample."""
    print(f"Processing {case_id}...")
    orig = load_original(case_id)
    conv = load_converted(case_id, split)

    # --- Subsample original point cloud ---
    pc_sub = subsample_points(orig["sensor_pc"], max_points=5000)

    # --- Get occupied voxel positions ---
    occ_indices = np.argwhere(conv["occupancy"] > 0.5)  # (N, 3)
    occ_world = voxel_to_world(occ_indices.astype(np.float64), conv["affine"])
    occ_sub_idx = subsample_points(np.arange(len(occ_world)), max_points=5000)
    occ_world_sub = occ_world[occ_sub_idx]

    # --- Figure 1: Point Cloud vs Occupancy Grid ---
    fig1 = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"Original Point Cloud ({len(orig['sensor_pc'])} pts, showing {len(pc_sub)})",
            f"Converted Occupancy Grid ({len(occ_indices)} voxels, showing {len(occ_world_sub)})"
        ],
    )

    fig1.add_trace(
        go.Scatter3d(
            x=pc_sub[:, 0], y=pc_sub[:, 1], z=pc_sub[:, 2],
            mode="markers",
            marker=dict(size=1, color="blue", opacity=0.5),
            name="Original PC"
        ),
        row=1, col=1
    )

    fig1.add_trace(
        go.Scatter3d(
            x=occ_world_sub[:, 0], y=occ_world_sub[:, 1], z=occ_world_sub[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="red", opacity=0.5),
            name="Occupancy Grid"
        ),
        row=1, col=2
    )

    fig1.update_layout(
        title=f"[{case_id}] Point Cloud vs Occupancy Grid",
        height=600, width=1200,
    )
    html_path1 = os.path.join(OUTPUT_DIR, f"{case_id}_pc_vs_occ.html")
    fig1.write_html(html_path1)
    print(f"  Saved: {html_path1}")

    # --- Figure 2: Label comparison (train only) ---
    if split == "train":
        orig_labels = orig["voxel_labels"]
        conv_labels = conv["labels"]

        # Get non-zero label positions
        orig_nonzero = np.argwhere(orig_labels > 0)  # anatomical structures only
        conv_nonzero = np.argwhere(conv_labels > 0)

        # Convert to world coordinates
        orig_world_min = orig["grid_world_min"]
        voxel_size = orig["grid_voxel_size"]

        orig_world = orig_world_min + orig_nonzero * voxel_size
        conv_label_world = voxel_to_world(conv_nonzero.astype(np.float64), conv["label_affine"])

        orig_world_sub = subsample_points(orig_world, max_points=5000)
        conv_label_world_sub = subsample_points(conv_label_world, max_points=5000)

        # Color by label
        orig_sub_idx = subsample_points(np.arange(len(orig_nonzero)), max_points=5000)
        conv_sub_idx = subsample_points(np.arange(len(conv_nonzero)), max_points=5000)
        orig_colors = orig_labels[orig_nonzero[orig_sub_idx, 0],
                                  orig_nonzero[orig_sub_idx, 1],
                                  orig_nonzero[orig_sub_idx, 2]]
        conv_colors = conv_labels[conv_nonzero[conv_sub_idx, 0],
                                  conv_nonzero[conv_sub_idx, 1],
                                  conv_nonzero[conv_sub_idx, 2]]

        fig2 = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            subplot_titles=[
                f"Original Labels (shape={orig_labels.shape}, {len(orig_nonzero)} nonzero)",
                f"Converted Labels (shape={conv_labels.shape}, {len(conv_nonzero)} nonzero)"
            ],
        )

        fig2.add_trace(
            go.Scatter3d(
                x=orig_world_sub[:, 0], y=orig_world_sub[:, 1], z=orig_world_sub[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=orig_colors, colorscale="Turbo",
                            cmin=0, cmax=69, opacity=0.5),
                name="Original Labels"
            ),
            row=1, col=1
        )

        fig2.add_trace(
            go.Scatter3d(
                x=conv_label_world_sub[:, 0], y=conv_label_world_sub[:, 1],
                z=conv_label_world_sub[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=conv_colors, colorscale="Turbo",
                            cmin=0, cmax=69, opacity=0.5),
                name="Converted Labels"
            ),
            row=1, col=2
        )

        fig2.update_layout(
            title=f"[{case_id}] Original vs Converted Labels",
            height=600, width=1200,
        )
        html_path2 = os.path.join(OUTPUT_DIR, f"{case_id}_labels.html")
        fig2.write_html(html_path2)
        print(f"  Saved: {html_path2}")

    # --- Print numerical verification ---
    print(f"\n  === Numerical Verification for {case_id} ===")
    print(f"  Original grid shape:    {orig['voxel_labels'].shape}")
    print(f"  Converted image shape:  {conv['occupancy'].shape}")
    print(f"  Target shape:           {TARGET_SHAPE}")
    print(f"  Shape match:            {conv['occupancy'].shape == TARGET_SHAPE}")
    print(f"  Occupancy density:      {conv['occupancy'].sum() / conv['occupancy'].size * 100:.3f}%")
    print(f"  Original PC points:     {len(orig['sensor_pc'])}")
    print(f"  Occupied voxels:        {int(conv['occupancy'].sum())}")
    print(f"  Affine diagonal:        {np.diag(conv['affine'])}")
    print(f"  Affine origin:          {conv['affine'][:3, 3]}")
    print(f"  grid_world_min:         {orig['grid_world_min']}")
    origin_match = np.allclose(conv["affine"][:3, 3], orig["grid_world_min"])
    print(f"  Origin matches:         {origin_match}")

    if split == "train":
        orig_lbl = orig["voxel_labels"]
        conv_lbl = conv["labels"]
        # Check that the original region is exactly preserved
        cx = min(orig_lbl.shape[0], TARGET_SHAPE[0])
        cy = min(orig_lbl.shape[1], TARGET_SHAPE[1])
        cz = min(orig_lbl.shape[2], TARGET_SHAPE[2])
        exact_match = np.all(conv_lbl[:cx, :cy, :cz] == orig_lbl[:cx, :cy, :cz])
        print(f"  Label content match:    {exact_match}")
        print(f"  Padded region all zero: {np.all(conv_lbl[cx:, :, :] == 0) and np.all(conv_lbl[:, cy:, :] == 0) and np.all(conv_lbl[:, :, cz:] == 0)}")
        orig_unique = np.unique(orig_lbl)
        conv_unique = np.unique(conv_lbl[:cx, :cy, :cz])
        print(f"  Original unique labels: {len(orig_unique)} classes: {orig_unique}")
        print(f"  Conv unique labels:     {len(conv_unique)} classes: {conv_unique}")

    return True


def create_statistics_summary():
    """Create a PNG chart showing dataset statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(SPLIT_JSON) as f:
        split = json.load(f)

    # Sample 20 random train samples for shape statistics
    rng = np.random.RandomState(42)
    sample_files = rng.choice(split["train"], size=min(100, len(split["train"])), replace=False)

    shapes = []
    densities = []
    point_counts = []

    for fname in sample_files:
        case_id = fname.replace(".npz", "")
        data = np.load(os.path.join(NPZ_DIR, fname))
        shapes.append(data["voxel_labels"].shape)
        point_counts.append(len(data["sensor_pc"]))

        # Load converted occupancy
        img_path = os.path.join(NNUNET_DIR, "imagesTr", f"{case_id}_0000.nii.gz")
        if os.path.exists(img_path):
            occ = nib.load(img_path).get_fdata()
            densities.append(occ.sum() / occ.size * 100)

    shapes = np.array(shapes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Grid size distributions
    for i, (dim, name) in enumerate(zip(shapes.T, ["X", "Y", "Z"])):
        axes[0, 0].hist(dim, bins=30, alpha=0.6, label=f"{name} [{dim.min()}-{dim.max()}]")
    axes[0, 0].set_title("Original Grid Size Distribution (100 samples)")
    axes[0, 0].set_xlabel("Voxels")
    axes[0, 0].legend()
    axes[0, 0].axvline(144, color="r", linestyle="--", alpha=0.5, label="Target X=144")
    axes[0, 0].axvline(128, color="g", linestyle="--", alpha=0.5, label="Target Y=128")

    # Occupancy density
    axes[0, 1].hist(densities, bins=30, color="steelblue", alpha=0.7)
    axes[0, 1].set_title(f"Occupancy Density (mean={np.mean(densities):.3f}%)")
    axes[0, 1].set_xlabel("Density (%)")

    # Point count distribution
    axes[1, 0].hist(point_counts, bins=30, color="coral", alpha=0.7)
    axes[1, 0].set_title(f"Point Cloud Size (mean={np.mean(point_counts):.0f})")
    axes[1, 0].set_xlabel("Number of points")

    # Padding analysis
    pad_x = TARGET_SHAPE[0] - shapes[:, 0]
    pad_y = TARGET_SHAPE[1] - shapes[:, 1]
    pad_z = TARGET_SHAPE[2] - shapes[:, 2]
    for pad, name in zip([pad_x, pad_y, pad_z], ["X", "Y", "Z"]):
        axes[1, 1].hist(pad, bins=30, alpha=0.6, label=f"{name} pad")
    axes[1, 1].set_title("Padding Amount per Dimension")
    axes[1, 1].set_xlabel("Voxels padded")
    axes[1, 1].legend()
    axes[1, 1].axvline(0, color="red", linestyle="--", alpha=0.5)

    plt.suptitle("HyperBody -> nnUNet Conversion Statistics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "conversion_statistics.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved statistics: {png_path}")


if __name__ == "__main__":
    # Select a few samples for visualization
    with open(SPLIT_JSON) as f:
        split = json.load(f)

    # Pick 3 train + 1 test sample
    train_samples = [
        split["train"][0].replace(".npz", ""),   # first train
        split["train"][100].replace(".npz", ""),  # middle train
    ]

    # Special case: BDMAP_00002911 (Y=129, clipping edge case)
    if "BDMAP_00002911.npz" in split["train"]:
        train_samples.append("BDMAP_00002911")

    test_sample = split["test"][0].replace(".npz", "")

    print("=" * 60)
    print("HyperBody -> nnUNet Conversion Verification")
    print("=" * 60)

    for case_id in train_samples:
        visualize_sample(case_id, split="train")
        print()

    visualize_sample(test_sample, split="test")
    print()

    # Create statistics summary
    print("\nGenerating statistics summary...")
    create_statistics_summary()

    print("\n" + "=" * 60)
    print("Verification complete! Check HTML files in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
