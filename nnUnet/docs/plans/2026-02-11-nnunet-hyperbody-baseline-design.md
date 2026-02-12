# nnUNet HyperBody Baseline Design

## Goal

Use nnUNet as a baseline for the HyperBody task: input partial body surface point cloud, predict full-body voxel segmentation (70 anatomical classes). Compare against our own method.

## Task Formulation

nnUNet is a 3D voxel image -> 3D voxel segmentation framework. It cannot process point clouds directly.

**Adaptation**: Voxelize `sensor_pc` into a single-channel binary occupancy grid (1 = point present, 0 = empty), then use standard nnUNet 3D segmentation pipeline.

- Input: 1-channel 3D binary occupancy grid (144, 128, 268)
- Output: 70-class voxel segmentation (144, 128, 268)
- Spacing: 4.0 x 4.0 x 4.0 mm (isotropic)

## Data Conversion

### Source Data

- Location: `Dataset/voxel_data/`, 10,779 `.npz` files
- Each file contains:
  - `sensor_pc`: (N, 3) float32 - partial surface point cloud
  - `voxel_labels`: (X, Y, Z) uint8 - 70-class ground truth
  - `grid_world_min/max`: (3,) float32 - world coordinate bounds
  - `grid_voxel_size`: [4.0, 4.0, 4.0] mm
  - `grid_occ_size`: (3,) int32 - grid dimensions
- Dynamic grid sizes (verified on all 10,779 samples):
  - X: 54 ~ 144, Y: 37 ~ 129, Z: 18 ~ 256
- Split: 9,779 train / 500 val / 500 test (from `dataset_split.json`)
- **Note**: 1 sample (BDMAP_00002911, Y=129) exceeds target Y=128 by 1 voxel; handled by clipping, consistent with our own method's behavior

### Target Format

```
nnUNet_data/nnUNet_raw/
└── Dataset501_HyperBody/
    ├── dataset.json
    ├── imagesTr/
    │   └── BDMAP_XXXXXXXX_0000.nii.gz    # occupancy grid, 9779 files
    ├── labelsTr/
    │   └── BDMAP_XXXXXXXX.nii.gz         # voxel_labels, 9779 files
    └── imagesTs/
        └── BDMAP_XXXXXXXX_0000.nii.gz    # occupancy grid, 500 files
```

### Conversion Logic (per sample)

Uses **corner-aligned padding** (matching our own method in `RUN/data/voxelizer.py`):

1. Load `.npz`, read `sensor_pc`, `voxel_labels`, `grid_world_min`, `grid_voxel_size`
2. Build occupancy grid directly at target size (clip indices to target bounds):
   ```python
   target_shape = (144, 128, 268)
   idx = np.floor((sensor_pc - grid_world_min) / grid_voxel_size).astype(np.int64)
   for d in range(3):
       idx[:, d] = np.clip(idx[:, d], 0, target_shape[d] - 1)
   occupancy = np.zeros(target_shape, dtype=np.float32)
   occupancy[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
   ```
3. Corner-pad labels to target size (crop if larger):
   ```python
   label = np.zeros(target_shape, dtype=np.uint8)
   x, y, z = voxel_labels.shape
   cx = min(x, target_shape[0])
   cy = min(y, target_shape[1])
   cz = min(z, target_shape[2])
   label[:cx, :cy, :cz] = voxel_labels[:cx, :cy, :cz]
   ```
4. Save as NIfTI with correct affine:
   ```python
   affine = np.diag([4.0, 4.0, 4.0, 1.0])
   affine[:3, 3] = grid_world_min  # origin = grid_world_min (corner-aligned)
   ```

### dataset.json

```json
{
    "channel_names": {
        "0": "noNorm"
    },
    "labels": {
        "background": 0,
        "liver": 1,
        "spleen": 2,
        "kidney_left": 3,
        "kidney_right": 4,
        "stomach": 5,
        "pancreas": 6,
        "gallbladder": 7,
        "urinary_bladder": 8,
        "prostate": 9,
        "heart": 10,
        "brain": 11,
        "thyroid_gland": 12,
        "spinal_cord": 13,
        "lung": 14,
        "esophagus": 15,
        "trachea": 16,
        "small_bowel": 17,
        "duodenum": 18,
        "colon": 19,
        "adrenal_gland_left": 20,
        "adrenal_gland_right": 21,
        "spine": 22,
        "rib_left_1": 23,
        "rib_left_2": 24,
        "rib_left_3": 25,
        "rib_left_4": 26,
        "rib_left_5": 27,
        "rib_left_6": 28,
        "rib_left_7": 29,
        "rib_left_8": 30,
        "rib_left_9": 31,
        "rib_left_10": 32,
        "rib_left_11": 33,
        "rib_left_12": 34,
        "rib_right_1": 35,
        "rib_right_2": 36,
        "rib_right_3": 37,
        "rib_right_4": 38,
        "rib_right_5": 39,
        "rib_right_6": 40,
        "rib_right_7": 41,
        "rib_right_8": 42,
        "rib_right_9": 43,
        "rib_right_10": 44,
        "rib_right_11": 45,
        "rib_right_12": 46,
        "skull": 47,
        "sternum": 48,
        "costal_cartilages": 49,
        "scapula_left": 50,
        "scapula_right": 51,
        "clavicula_left": 52,
        "clavicula_right": 53,
        "humerus_left": 54,
        "humerus_right": 55,
        "hip_left": 56,
        "hip_right": 57,
        "femur_left": 58,
        "femur_right": 59,
        "gluteus_maximus_left": 60,
        "gluteus_maximus_right": 61,
        "gluteus_medius_left": 62,
        "gluteus_medius_right": 63,
        "gluteus_minimus_left": 64,
        "gluteus_minimus_right": 65,
        "autochthon_left": 66,
        "autochthon_right": 67,
        "iliopsoas_left": 68,
        "iliopsoas_right": 69
    },
    "numTraining": 9779,
    "file_ending": ".nii.gz"
}
```

## Environment Setup

```bash
# Step 0: Install nnUNet and all dependencies
conda activate nnunet
pip install -e /home/comp/csrkzhu/code/Compare/nnUNet

# Step 1: Set environment variables
export nnUNet_raw="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_results"
```

## Training Pipeline

```bash
# Step 1: Verify dataset and plan experiments
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity

# Step 2: Train 5-fold cross-validation (3d_fullres)
nnUNetv2_train 501 3d_fullres 0 --npz
nnUNetv2_train 501 3d_fullres 1 --npz
nnUNetv2_train 501 3d_fullres 2 --npz
nnUNetv2_train 501 3d_fullres 3 --npz
nnUNetv2_train 501 3d_fullres 4 --npz

# Step 3: Find best configuration
nnUNetv2_find_best_configuration 501 -c 3d_fullres

# Step 4: Inference on test set
nnUNetv2_predict -i $nnUNet_raw/Dataset501_HyperBody/imagesTs \
                 -o $nnUNet_results/Dataset501_HyperBody/predictions \
                 -d 501 -c 3d_fullres
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input representation | Binary occupancy grid | Simplest point cloud -> voxel conversion; matches our method's input for fair comparison |
| Number of channels | 1 (noNorm) | Binary input; `NoNormalization` preserves 0/1 values as-is |
| Volume size | (144, 128, 268) | Match our own method's `config.volume_size` for fair comparison |
| Padding strategy | Corner-aligned | Match our own method (`RUN/data/voxelizer.py`); body origin at grid corner |
| nnUNet configuration | 3d_fullres | Native 3D data, no need for 2D or cascade |
| Dataset ID | 501 | Avoid conflict with existing nnUNet datasets |

## Known Limitations and Impact

### 1. Sparse occupancy vs nnUNet's dense image assumption (significant)

nnUNet's `crop_to_nonzero` (`cropping.py`) creates a nonzero mask from the input image and applies `binary_fill_holes`. For dense medical images (CT/MRI), this fills the body interior. However, our binary occupancy grid is **extremely sparse (~1% non-zero)** — only surface points, not a closed surface. `binary_fill_holes` has no effect.

**Consequence**: In `crop_to_nonzero` line 36:
```python
seg[(seg == 0) & (~nonzero_mask)] = nonzero_label  # nonzero_label = -1
```
Label-0 voxels ("inside_body_empty") NOT at surface point locations are marked as -1 (ignore). This means:
- All non-zero class voxels (organs, bones, muscles) are **unaffected** — these are learned normally
- Class 0 ("inside_body_empty") is mostly ignored during training — only class-0 voxels coinciding with surface points contribute to the loss
- nnUNet will still implicitly predict class 0 as default (majority class), but with weaker supervision

**Accepted**: This is an inherent mismatch between sparse binary input and nnUNet's design. The 69 anatomical classes remain fully supervised, making this acceptable for a baseline comparison.

### 2. Other limitations

- Binary occupancy loses fine-grained geometric information (point density, local normals)
- 70 classes with highly imbalanced distribution (73.8% background) may challenge nnUNet's default loss weighting
- nnUNet preprocessing crops to non-zero bounding box, effectively removing padding (padding is still required for valid NIfTI spatial reference)
- 500 val samples from `dataset_split.json` are excluded; nnUNet uses its own 5-fold CV on the 9,779 training samples
- 1 sample (BDMAP_00002911) is clipped by 1 voxel in Y dimension — negligible impact

## Implementation Steps

1. Install nnUNet: `conda activate nnunet && pip install -e .`
2. Write data conversion script (`convert_hyperbody_to_nnunet.py`)
3. Run conversion for train + test splits
4. Set environment variables
5. Run `nnUNetv2_plan_and_preprocess`
6. Train 3d_fullres (5 folds)
7. Evaluate and compare with our method

---

## Progress Log

### Step 1: nnUNet Installation [DONE]

- Environment: `conda activate nnunet` (Python 3.10)
- Installed: `pip install -e /home/comp/csrkzhu/code/Compare/nnUNet`
- nnUNet v2.6.4 with all dependencies

### Step 2: Data Conversion Script [DONE]

**Files created:**

| File | Description |
|------|-------------|
| `convert_hyperbody_to_nnunet.py` | Main conversion script, 8-worker parallel processing |
| `test_convert_hyperbody_to_nnunet.py` | TDD test suite, 18 tests |
| `visualize_conversion.py` | Interactive Plotly verification visualizations |

**TDD test results:** 18/18 passed

```
TestBuildOccupancyGrid      (5 tests) - shape, placement, duplicates, clipping, negative indices
TestPadLabels               (4 tests) - smaller, exact, larger (clip), class preservation
TestCreateNiftiAffine       (1 test)  - diagonal spacing + origin
TestConvertSingleSample     (3 tests) - file creation, content correctness, oversized clipping
TestDatasetJson             (2 tests) - fields, consecutive labels
TestFileNaming              (2 tests) - naming convention
TestRealDataSanity          (1 test)  - end-to-end on real BDMAP_00000001.npz
```

### Step 3: Data Conversion [DONE]

**Conversion output:**

| Directory | File Count | Content |
|-----------|-----------|---------|
| `nnUNet_data/nnUNet_raw/Dataset501_HyperBody/imagesTr/` | 9,779 | `*_0000.nii.gz` occupancy grids |
| `nnUNet_data/nnUNet_raw/Dataset501_HyperBody/labelsTr/` | 9,779 | `*.nii.gz` voxel labels |
| `nnUNet_data/nnUNet_raw/Dataset501_HyperBody/imagesTs/` | 500 | `*_0000.nii.gz` occupancy grids |
| `nnUNet_data/nnUNet_raw/Dataset501_HyperBody/dataset.json` | 1 | 70 labels, noNorm, .nii.gz |

**Conversion failures:** 0

**Verification results (4 samples including BDMAP_00002911 edge case):**

| Check | Result |
|-------|--------|
| Output shape = (144, 128, 268) | All True |
| Affine origin = grid_world_min | All True |
| Spacing = 4.0mm isotropic | All True |
| Binary occupancy (only 0/1 values) | All True |
| Label content exactly preserved | All True |
| Padded regions all zero | All True |
| BDMAP_00002911 Y=129 clipped to 128 | True |

**Dataset statistics (from 100 sampled files):**
- Mean occupancy density: 0.295% (extremely sparse, as expected)
- Mean point cloud size: ~60,767 points
- Original grid size ranges: X [75-129], Y [56-106], Z [31-249]

Verification visualizations saved to `docs/verification/`:
- `BDMAP_*_pc_vs_occ.html` — Point cloud vs occupancy grid (interactive 3D)
- `BDMAP_*_labels.html` — Original vs converted labels (interactive 3D)
- `conversion_statistics.png` — Distribution charts

### Step 4: Environment Variables [DONE]

```bash
export nnUNet_raw="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_results"
```

### Step 5: nnUNet Plan and Preprocess [DONE]

**Command:** `nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity`

**Dataset integrity:** Passed

**nnUNet auto-planned configurations:**

| Parameter | 3d_fullres | 2d |
|-----------|-----------|-----|
| Architecture | PlainConvUNet (Conv3d) | PlainConvUNet (Conv2d) |
| Stages | 5 | 5 |
| Features | [32, 64, 128, 256, 320] | [32, 64, 128, 256, 512] |
| Patch size | [128, 64, 112] | [64, 112] |
| Batch size | 4 | 380 |
| Normalization | NoNormalization | NoNormalization |
| Spacing | [4.0, 4.0, 4.0] | [4.0, 4.0] |
| batch_dice | False | True |

**Median image size after crop:** [114, 64, 98] (nnUNet cropped to non-zero bounding box as expected)

**Foreground intensity:** max=1.0, mean=0.0005, confirming sparse binary input is handled correctly.

**Preprocessed output:**
- `nnUNetPlans_2d/`: 29,337 files (9,779 samples × 3 files each)
- `nnUNetPlans_3d_fullres/`: 29,337 files
- `gt_segmentations/`: ground truth segmentations
- `nnUNetPlans.json`: experiment plans
- `dataset_fingerprint.json`: dataset statistics

### Step 5.5: Test Set Ground Truth Labels [DONE]

Created `labelsTs/` (500 NIfTI files) for evaluation after inference.

**Files modified:**

| File | Changes |
|------|---------|
| `convert_hyperbody_to_nnunet.py` | Added `convert_test_label()`, `--create_test_labels` CLI flag |
| `test_convert_hyperbody_to_nnunet.py` | Added 8 new tests (26 total, all pass) |

**Test results:** 26/26 passed

### Step 6: Training [IN PROGRESS]

**Issue encountered:** `RuntimeError: One or more background workers are no longer alive` with multi-threaded data augmentation. Root cause: Python multiprocessing `fork` start method + CUDA context inheritance conflict.

**Initial workaround:** `nnUNet_n_proc_DA=0` (single-threaded, ~11 min/epoch).

**Proper fix:** Added `multiprocessing.set_start_method('spawn', force=True)` to `nnunetv2/run/run_training.py` (in both `run_training_entry()` and `__main__` block). The nnUNet developers had this line commented out at line 277 — we enabled it. This allows full multi-threaded data augmentation (default 12 workers) without crashes.

**Speedup: 10x** — epoch time reduced from ~630s to ~62s. Estimated total per-fold time: **~17 hours** (was ~7.7 days).

**Active training:**

| Fold | GPU | Resumed From | Status | Notes |
|------|-----|------------|--------|-------|
| 0 | GPU 0 (RTX 4090) | Epoch 50 | Training | ~62 sec/epoch, ~17h remaining |
| 1 | GPU 1 (RTX 4090) | Epoch 50 | Training | ~65 sec/epoch, ~17h remaining |
| 2-4 | — | — | Pending | Start after 0+1 complete |

**Commands used:**
```bash
# Fold 0 (GPU 0) - resume from checkpoint
nnUNetv2_train 501 3d_fullres 0 --npz --c

# Fold 1 (GPU 1) - resume from checkpoint
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 501 3d_fullres 1 --npz --c
```

**Training logs:**
- Fold 0: `nnUNet_data/nnUNet_results/Dataset501_HyperBody/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2026_2_12_00_08_12.txt`
- Fold 1: `nnUNet_data/nnUNet_results/Dataset501_HyperBody/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/training_log_2026_2_12_00_23_04.txt`

**Epoch 0 metrics (fold 0):** Train loss 1.1976, Val loss 0.8422, Pseudo Dice 0.0 (expected at start)

### Step 7: Evaluation Script [DONE]

**Files created:**

| File | Description |
|------|-------------|
| `evaluate_nnunet_predictions.py` | Evaluation pipeline (~350 lines), vectorized Dice computation |
| `test_evaluate_nnunet_predictions.py` | TDD test suite, 33 tests |

**TDD test results:** 33/33 passed

```
TestComputeDicePerClass      (9 tests)  - perfect match, no overlap, partial overlap, absent class, gt-only, pred-only, all 70, vectorization, background
TestComputeMeanDice          (5 tests)  - exclude/include background, skip NaN, all NaN, single class
TestComputeGroupDice         (5 tests)  - all groups, organ/muscle composition, NaN handling
TestEvaluateSingleSample     (3 tests)  - perfect prediction, wrong prediction, returns all classes
TestEvaluateOriginalSpace    (2 tests)  - crop before Dice, padded vs original difference
TestSaveResults              (3 tests)  - JSON format, per-case structure, mean_dice type
TestPlotPerClassDice         (2 tests)  - creates PNG, handles NaN
TestEvaluateFolder           (2 tests)  - matches files, skips unmatched
TestRealDataIntegration      (2 tests)  - GT self-evaluation (padded + original space)
```

**Usage (after training completes):**
```bash
# Run inference
nnUNetv2_predict -i $nnUNet_raw/Dataset501_HyperBody/imagesTs \
                 -o $nnUNet_results/Dataset501_HyperBody/predictions \
                 -d 501 -c 3d_fullres

# Evaluate (padded space)
python evaluate_nnunet_predictions.py \
    --pred_dir $nnUNet_results/Dataset501_HyperBody/predictions/ \
    --gt_dir $nnUNet_raw/Dataset501_HyperBody/labelsTs/ \
    --output_dir evaluation_results/

# Evaluate (original space)
python evaluate_nnunet_predictions.py \
    --pred_dir $nnUNet_results/Dataset501_HyperBody/predictions/ \
    --gt_dir $nnUNet_raw/Dataset501_HyperBody/labelsTs/ \
    --output_dir evaluation_results/ \
    --npz_dir Dataset/voxel_data/ \
    --original_space
```

**Organ groups for reporting:**

| Group | Classes | Count |
|-------|---------|-------|
| Organs | 1-12 (liver, spleen, kidneys, stomach, pancreas, gallbladder, bladder, prostate, heart, brain, thyroid) | 12 |
| Soft tissue | 13-21 (spinal cord, lung, esophagus, trachea, bowels, colon, adrenal glands) | 9 |
| Bones | 22-59 (spine, ribs, skull, sternum, cartilages, scapulae, claviculae, humeri, hips, femurs) | 38 |
| Muscles | 60-69 (gluteus, autochthon, iliopsoas) | 10 |

### Step 8: Post-Training Pipeline [PENDING]

After all 5 folds complete:
```bash
# Find best configuration
nnUNetv2_find_best_configuration 501 -c 3d_fullres

# Inference on test set
nnUNetv2_predict -i $nnUNet_raw/Dataset501_HyperBody/imagesTs \
                 -o $nnUNet_results/Dataset501_HyperBody/predictions \
                 -d 501 -c 3d_fullres

# Evaluate
python evaluate_nnunet_predictions.py \
    --pred_dir $nnUNet_results/Dataset501_HyperBody/predictions/ \
    --gt_dir $nnUNet_raw/Dataset501_HyperBody/labelsTs/ \
    --output_dir evaluation_results/ \
    --npz_dir Dataset/voxel_data/ \
    --original_space
```
