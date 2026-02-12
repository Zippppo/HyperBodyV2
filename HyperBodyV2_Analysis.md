# HyperBodyV2 Integration Analysis

> HyperbodyV1 (Hyperbolic Module) + nnUnet (Backbone) = HyperBodyV2

---

## 1. Project Overview

| Project | Path | Conda Env | Description |
|---------|------|-----------|-------------|
| HyperbodyV1 | `HyperbodyV1/` | pasco | 3D UNet + Hyperbolic module |
| nnUnet | `nnUnet/` | nnunet | PlainConvUNet, adapted for HyperBody dataset |

Both projects solve the same task: **input partial human body surface point cloud, predict full body voxel labels (70 classes)**.

---

## 2. HyperbodyV1 Architecture

### 2.1 BodyNet (Top-Level Model)

**File:** `HyperbodyV1/models/body_net.py` (115 lines)

```python
class BodyNet(nn.Module):
    # Combines UNet3D + LorentzProjectionHead + LorentzLabelEmbedding
    def forward(x):
        logits, d2 = self.unet(x, return_features=True)
        voxel_emb = self.hyp_head(d2)      # (B, 32, D, H, W) in Lorentz space
        label_emb = self.label_emb()        # (70, 32) in Lorentz space
        return logits, voxel_emb, label_emb
```

### 2.2 UNet3D Backbone

**File:** `HyperbodyV1/models/unet3d.py` (138 lines)

- **Input:** `(B, 1, D, H, W)` binary occupancy grid, full volume `(B, 1, 144, 128, 268)`
- **Output:** `logits (B, 70, D, H, W)` + `d2 (B, 32, D, H, W)`

**Channel Progression:**

| Layer | Channels | Spatial |
|-------|----------|---------|
| enc1 | 1 -> 32 | D, H, W |
| enc2 | 32 -> 64 | D/2, H/2, W/2 |
| enc3 | 64 -> 128 | D/4, H/4, W/4 |
| enc4 | 128 -> 256 | D/8, H/8, W/8 |
| bottleneck (DenseBlock) | 256 -> 384 | D/8, H/8, W/8 |
| dec4 | 384+128 -> 128 | D/4, H/4, W/4 |
| dec3 | 128+64 -> 64 | D/2, H/2, W/2 |
| **dec2** | **64+32 -> 32** | **D, H, W** |
| final (1x1x1 conv) | 32 -> 70 | D, H, W |

**Key:** `dec2` output `(B, 32, D, H, W)` is the interface to the Hyperbolic module.

### 2.3 Hyperbolic Module

All hyperbolic components are in `HyperbodyV1/models/hyperbolic/`.

#### LorentzProjectionHead

**File:** `HyperbodyV1/models/hyperbolic/projection_head.py` (59 lines)

```
Input: (B, 32, D, H, W) from decoder
  -> Conv3d 1x1x1 (32 -> embed_dim=32)
  -> exp_map0 (tangent space -> Lorentz manifold)
Output: (B, 32, D, H, W) in Lorentz space
```

- `in_channels=32`, `embed_dim=32`, `curv=1.0`
- Just a 1x1x1 conv followed by exponential map

#### LorentzLabelEmbedding

**File:** `HyperbodyV1/models/hyperbolic/label_embedding.py` (218 lines)

```
No input (learnable parameters)
Output: (num_classes=70, embed_dim=32) in Lorentz space
```

- Learnable tangent vectors mapped to Lorentz manifold via `exp_map0`
- Initialization: hierarchy-aware (depth-based norms, direction from text embeddings or random)
- `direction_mode`: "random" or "semantic" (using pre-computed text embeddings)
- `class_depths`: dict mapping class_id -> hierarchy depth
- `min_radius=0.1`, `max_radius=2.0`

#### Lorentz Operations

**File:** `HyperbodyV1/models/hyperbolic/lorentz_ops.py` (169 lines)

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `exp_map0(v, curv)` | `[..., D]` tangent | `[..., D]` manifold | Tangent -> Lorentz |
| `log_map0(x, curv)` | `[..., D]` manifold | `[..., D]` tangent | Lorentz -> Tangent |
| `pointwise_dist(x, y, curv)` | `[..., D]`, `[..., D]` | `[...]` | Element-wise geodesic distance |
| `pairwise_dist(x, y, curv)` | `[N, D]`, `[M, D]` | `[N, M]` | All-pairs distance matrix |
| `distance_to_origin(x, curv)` | `[..., D]` | `[...]` | Distance from origin |
| `lorentz_to_poincare(x)` | `[..., D]` | `[..., D-1]` | Visualization conversion |

#### LorentzRankingLoss

**File:** `HyperbodyV1/models/hyperbolic/lorentz_loss.py` (471 lines)

**Class `LorentzRankingLoss` (Lines 32-242):**

```python
def forward(voxel_emb, labels, label_emb):
    # voxel_emb: (B, 32, D, H, W)  - Lorentz embeddings
    # labels:    (B, D, H, W)       - ground truth
    # label_emb: (70, 32)           - class embeddings
    # Returns: scalar loss
```

Algorithm:
1. Flatten voxel_emb -> (N, 32), N = B*D*H*W
2. Stratified sampling: up to `num_samples_per_class=64` voxels per class
3. Positive distance: `d_pos = pointwise_dist(anchor, label_emb[true_class])`
4. Negative sampling (curriculum): warmup=uniform, then temperature-decay hard mining
5. Triplet loss: `max(0, margin + d_pos - d_neg)`, `margin=0.4`

**Class `LorentzTreeRankingLoss` (Lines 245-471):**
- Same as above, but uses tree/graph distance for negative sampling weights
- Requires `tree_dist_matrix: (70, 70)`

### 2.4 Loss & Training

**Loss:** `HyperbodyV1/models/losses.py` (263 lines)
- `CombinedLoss` = 0.5 * CE + 0.5 * Dice

**Training:** `HyperbodyV1/train.py` (600+ lines)
```python
total_loss = seg_loss + hyp_weight * hyp_loss  # hyp_weight = 0.05
```

**Config:** `HyperbodyV1/config.py` (136 lines)

Key hyperbolic hyperparameters:
```
hyp_embed_dim: 32
hyp_curv: 1.0
hyp_margin: 0.4
hyp_samples_per_class: 64
hyp_num_negatives: 8
hyp_t_start: 2.0       # temperature start (uniform sampling)
hyp_t_end: 0.1         # temperature end (hard mining)
hyp_warmup_epochs: 6
hyp_weight: 0.05       # loss weight
hyp_freeze_epochs: 5   # freeze label embeddings first N epochs
hyp_text_lr_ratio: 0.01
hyp_text_grad_clip: 0.1
hyp_distance_mode: "graph"  # tree / graph / hyperbolic
```

Training details:
- Optimizer: AdamW, visual params lr=1e-3, text params lr=1e-5
- AMP enabled, forces FP32 for hyperbolic distance computation
- Gradient clipping: 1.0 (standard), 0.1 (text params on first unfreeze)

---

## 3. nnUnet Architecture

### 3.1 PlainConvUNet Backbone

**Network build:** `nnUnet/nnunetv2/utilities/get_network_from_plans.py` (Lines 9-43)

**Network class:** `dynamic_network_architectures.architectures.unet.PlainConvUNet`

**Plans config:** `nnUnet/nnUNet_data/nnUNet_preprocessed/Dataset501_HyperBody/nnUNetPlans.json`

**3d_fullres configuration:**
```
n_stages: 5
features_per_stage: [32, 64, 128, 256, 320]
kernel_sizes: [[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]]
strides: [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]
n_conv_per_stage: [2, 2, 2, 2, 2]
n_conv_per_stage_decoder: [2, 2, 2, 2]
norm: InstanceNorm3d
activation: LeakyReLU (inplace)
```

- **Input:** `(B, 1, 128, 64, 112)` - binary occupancy patch
- **Output:** `(B, 70, 128, 64, 112)` - logits
- **Deep supervision:** enabled (multi-scale output list during training)
- **Batch size:** 4

**Channel Progression:**

| Stage | Encoder Channels | Decoder Channels | Spatial (relative) |
|-------|------------------|------------------|-------------------|
| 0 | 1 -> 32 | 32 | Full res |
| 1 | 32 -> 64 | 64 -> 32 | 1/2 |
| 2 | 64 -> 128 | 128 -> 64 | 1/4 |
| 3 | 128 -> 256 | 256 -> 128 | 1/8 |
| 4 (bottleneck) | 256 -> 320 | - | 1/16 |

**Decoder output at final stage: (B, 32, D, H, W)** -> seg_head (1x1x1 conv) -> (B, 70, D, H, W)

### 3.2 Trainer

**File:** `nnUnet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` (1387+ lines)

Key methods:
| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | 71-198 | Initialization |
| `initialize` | 200-238 | Build network, optimizer, loss, dataloaders |
| `build_network_architecture` | 304-336 | Instantiate PlainConvUNet from plans |
| `_build_loss` | 391-425 | DC_and_CE_loss + DeepSupervision wrapper |
| `configure_optimizers` | 507-511 | SGD + PolyLR |
| `train_step` | 973-1003 | Forward + loss + backward |
| `validation_step` | 1020-1085 | Validation with Dice metrics |
| `run_training` | 1366-1387 | Main training loop |

### 3.3 Loss Function

**File:** `nnUnet/nnunetv2/training/loss/compound_losses.py` (Lines 8-56)

```python
DC_and_CE_loss:
    weight_ce=1, weight_dice=1
    dice_class = MemoryEfficientSoftDiceLoss (do_bg=False)
    ce_class = RobustCrossEntropyLoss
```

Wrapped in `DeepSupervisionWrapper` (exponentially decreasing weights per scale).

**Dice Loss:** `nnUnet/nnunetv2/training/loss/dice.py` (Lines 58-96)
**Deep Supervision:** `nnUnet/nnunetv2/training/loss/deep_supervision.py` (Lines 1-30)

### 3.4 Optimizer & Scheduler

```
SGD(lr=0.01, weight_decay=3e-5, momentum=0.99, nesterov=True)
PolyLRScheduler (polynomial decay)
Gradient clipping: max_norm=12
```

### 3.5 Data Pipeline

**Conversion script:** `nnUnet/convert_hyperbody_to_nnunet.py` (Lines 25-54)
- Target volume shape: `(144, 128, 268)`
- Voxel size: 4.0mm isotropic
- 70 classes (background + 69 organs/tissues/bones/muscles)
- 10,279 training samples

**Dataset config:** `nnUnet/nnUNet_data/nnUNet_raw/Dataset501_HyperBody/dataset.json`

**Data loader:** `nnUnet/nnunetv2/training/dataloading/data_loader.py` (Lines 19-89)
- Patch-based sampling: `(128, 64, 112)`
- Foreground oversampling: 33%

**Evaluation:** `nnUnet/evaluate_nnunet_predictions.py`
- Per-class Dice, mean Dice (excluding background), group metrics

---

## 4. Integration Feasibility

### 4.1 Interface Match

```
HyperbodyV1:  UNet3D dec2 output   (B, 32, D, H, W)  ->  LorentzProjectionHead
nnUnet:       PlainConvUNet decoder (B, 32, D, H, W)  ->  seg_head (1x1x1 conv)
```

Both decoders produce **32-channel feature maps at full resolution** before the final classification head. The `LorentzProjectionHead` can directly accept nnUnet's decoder features without any dimension adaptation.

### 4.2 HyperBodyV2 Architecture Sketch

```
PlainConvUNet (nnUnet backbone, pretrained weights available)
    |
    +-- decoder_features (B, 32, D, H, W)
    |       |
    |       +-> LorentzProjectionHead (32 -> 32)
    |       |       |
    |       |       +-> voxel_emb (B, 32, D, H, W) in Lorentz space
    |       |
    |       +-> LorentzLabelEmbedding
    |               |
    |               +-> label_emb (70, 32) in Lorentz space
    |
    +-- logits (B, 70, D, H, W)  [+ deep supervision outputs]
            |
            +-> DC_and_CE_loss (with DeepSupervision)

Loss = seg_loss + hyp_weight * LorentzRankingLoss(voxel_emb, labels, label_emb)
```

### 4.3 Issues to Resolve

| Issue | Description | Difficulty |
|-------|-------------|------------|
| **A. Feature extraction** | PlainConvUNet forward only returns logits; need to extract decoder features before seg_head | Medium - subclass or hook |
| **B. Patch vs full volume** | nnUnet trains on patches (128,64,112), not full volumes (144,128,268). Hyperbolic loss sampling per patch will see fewer classes | Low - LorentzRankingLoss already handles variable class counts |
| **C. Deep supervision** | Hyperbolic loss only needs full-res features; deep supervision is independent | Low - no conflict |
| **D. Optimizer groups** | Need separate param groups for backbone (SGD) vs hyperbolic (lower LR) | Low - standard PyTorch |
| **E. AMP compatibility** | Hyperbolic distance ops need FP32; nnUnet uses GradScaler | Low - already solved in HyperbodyV1 |
| **F. Pretrained init** | Can load nnUnet trained weights for backbone; hyperbolic params train from scratch | Low - partial state_dict loading |

### 4.4 Implementation Strategy

The cleanest approach is to create a **custom nnUNetTrainer subclass** (e.g., `nnUNetTrainer_HyperBody`):

1. Override `build_network_architecture()` to wrap PlainConvUNet and expose decoder features
2. Override `_build_loss()` to add LorentzRankingLoss alongside DC_and_CE_loss
3. Override `configure_optimizers()` to add param groups for hyperbolic module
4. Override `train_step()` to compute combined loss
5. Copy hyperbolic module files from HyperbodyV1 into nnUnet project

This keeps the nnUnet framework intact and all existing tooling (inference, evaluation, etc.) working.

---

## 5. Key File Index

### HyperbodyV1

| Component | Path |
|-----------|------|
| Top-level model | `HyperbodyV1/models/body_net.py` |
| 3D UNet | `HyperbodyV1/models/unet3d.py` |
| DenseBlock | `HyperbodyV1/models/dense_block.py` |
| Projection head | `HyperbodyV1/models/hyperbolic/projection_head.py` |
| Label embedding | `HyperbodyV1/models/hyperbolic/label_embedding.py` |
| Lorentz ops | `HyperbodyV1/models/hyperbolic/lorentz_ops.py` |
| Lorentz loss | `HyperbodyV1/models/hyperbolic/lorentz_loss.py` |
| Seg losses (CE+Dice) | `HyperbodyV1/models/losses.py` |
| Training script | `HyperbodyV1/train.py` |
| Config dataclass | `HyperbodyV1/config.py` |
| Dataset | `HyperbodyV1/data/dataset.py` |
| Example config YAML | `HyperbodyV1/configs/LR-SOTA.yaml` |
| Tests | `HyperbodyV1/tests/` |

### nnUnet

| Component | Path |
|-----------|------|
| Trainer (core) | `nnUnet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` |
| Network builder | `nnUnet/nnunetv2/utilities/get_network_from_plans.py` |
| Plans handler | `nnUnet/nnunetv2/utilities/plans_handling/plans_handler.py` |
| Compound loss | `nnUnet/nnunetv2/training/loss/compound_losses.py` |
| Dice loss | `nnUnet/nnunetv2/training/loss/dice.py` |
| Deep supervision loss | `nnUnet/nnunetv2/training/loss/deep_supervision.py` |
| Label manager | `nnUnet/nnunetv2/utilities/label_handling/label_handling.py` |
| Data loader | `nnUnet/nnunetv2/training/dataloading/data_loader.py` |
| Dataset class | `nnUnet/nnunetv2/training/dataloading/nnunet_dataset.py` |
| Inference | `nnUnet/nnunetv2/inference/predict_from_raw_data.py` |
| LR scheduler | `nnUnet/nnunetv2/training/lr_scheduler/polylr.py` |
| Plans JSON | `nnUnet/nnUNet_data/nnUNet_preprocessed/Dataset501_HyperBody/nnUNetPlans.json` |
| Dataset JSON | `nnUnet/nnUNet_data/nnUNet_raw/Dataset501_HyperBody/dataset.json` |
| Data conversion | `nnUnet/convert_hyperbody_to_nnunet.py` |
| Evaluation | `nnUnet/evaluate_nnunet_predictions.py` |
