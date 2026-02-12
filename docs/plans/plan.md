# HyperBodyV2 Implementation Plan

## Context

HyperBodyV2 = nnUnet PlainConvUNet backbone + HyperbodyV1 Hyperbolic Module.

Both projects solve the same task (input partial body surface point cloud -> predict full body voxel labels, 70 classes). Both decoders produce `(B, 32, D, H, W)` features at full resolution before the final classification head. The goal is to graft HyperbodyV1's Lorentz-space embedding branch onto nnUnet's stronger backbone while preserving the entire nnUnet framework (training, inference, evaluation).

**Key interface match:** PlainConvUNet decoder final stage outputs `(B, 32, D, H, W)` -> `LorentzProjectionHead(in_channels=32)` accepts this directly.

---

## Design Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Feature extraction | **Forward hook** on `decoder.stages[-1]` | Zero modification to PlainConvUNet. Validate with gradient flow test first; if `torch.compile` breaks hooks, disable compile in trainer. |
| Training strategy | **From scratch** | No pretrained weight loading. Keep `load_pretrained_backbone()` as optional utility. |
| Optimizer | **Split**: SGD for backbone, AdamW for hyperbolic | SGD(lr=0.01, momentum=0.99) for backbone (proven for nnUnet). AdamW(lr=1e-3) for hyp_head, AdamW(lr=1e-5) for label_emb (better for embeddings). Two optimizers, two schedulers. |
| Distance mode | **Graph** | Precomputed `Dataset/graph_distance_matrix.pt` |
| Direction mode | **Random** | No text embedding file needed |
| Validation hyp_loss | **Yes** | Compute hyp_loss during validation for monitoring |
| Logging | **nnUnet logger only** | Add seg_loss + hyp_loss to existing `print_to_log_file` |
| File location | **Inside nnUnet** | `variants/hyperbolic/` dir, auto-discovered by nnUnet CLI |

---

## Architecture Overview

```
PlainConvUNet backbone (trained from scratch)
    |
    +-- encoder(x) -> skips
    +-- decoder(skips):
            |
            +-- stages[0..3] -> transpconv + cat skip + StackedConvBlocks
            |       |
            |       +-- stages[-1] output: decoder_features (B, 32, D, H, W)
            |               |                                [captured by hook]
            |               +-> LorentzProjectionHead (32->32, 1x1 conv + exp_map0)
            |               |       -> voxel_emb (B, 32, D, H, W) in Lorentz space
            |               |
            |               +-> seg_layers[-1] (1x1 conv, 32->70)
            |                       -> logits (B, 70, D, H, W)
            |
            +-- LorentzLabelEmbedding (learnable, hierarchy-aware)
                    -> label_emb (70, 32) in Lorentz space

Optimizers:
  SGD  -> backbone params           (lr=0.01, momentum=0.99, nesterov=True)
  AdamW -> hyp_head + label_emb     (lr=1e-3 / 1e-5 respectively)

Loss = DC_and_CE_loss(logits, target) [+ DeepSupervision]
     + 0.05 * LorentzRankingLoss(voxel_emb, labels, label_emb)
```

---

## File Structure

All new files under `nnUnet/nnunetv2/training/nnUNetTrainer/variants/hyperbolic/`:

| File | Source | Description |
|------|--------|-------------|
| `__init__.py` | New | Package init, re-exports |
| `nnUNetTrainerHyperBody.py` | New | Trainer subclass |
| `hyperbolic_network.py` | New | `HyperBodyNet` wrapper model |
| `lorentz_ops.py` | Copy from `HyperbodyV1/models/hyperbolic/lorentz_ops.py` | No import changes needed |
| `projection_head.py` | Copy from `HyperbodyV1/models/hyperbolic/projection_head.py` | Fix: `from .lorentz_ops` |
| `label_embedding.py` | Copy from `HyperbodyV1/models/hyperbolic/label_embedding.py` | Fix: `from .lorentz_ops` |
| `lorentz_loss.py` | Copy from `HyperbodyV1/models/hyperbolic/lorentz_loss.py` | Fix: `from .lorentz_ops` |
| `organ_hierarchy.py` | Copy from `HyperbodyV1/data/organ_hierarchy.py` | No import changes needed |

Tests at `nnUnet/tests/hyperbolic/`:

| File | Description |
|------|-------------|
| `test_hook_gradient_flow.py` | **FIRST TEST** - Validate hook approach: hook captures features, gradients flow through hook to encoder, test with and without torch.compile |
| `test_lorentz_ops.py` | Verify copied ops: shapes, NaN-free, inverse consistency |
| `test_hyperbolic_network.py` | Wrapper model: training mode tuple, inference mode tensor, state_dict |
| `test_trainer_hyperbody.py` | Trainer: dual optimizers via `configure_optimizers` (backbone returned, hyp stored as attr), both LR schedulers step, loss build, freeze/unfreeze, curriculum, checkpoint save/load round-trip |
| `test_integration.py` | End-to-end: train_step backward, AMP, gradient flow |

---

## Step-by-Step Implementation

### Step 0: Validate hook approach (TDD gate)

**Test:** `test_hook_gradient_flow.py`

Before any implementation, validate the core assumption:
1. Instantiate `PlainConvUNet` (5 stages, features=[32,64,128,256,320])
2. Register forward hook on `decoder.stages[-1]`
3. Synthetic input `(1, 1, 32, 16, 32)` -> forward -> hook captures `(1, 32, 32, 16, 32)` features
4. Compute `hooked_features.sum().backward()`
5. Assert encoder params have non-zero `.grad`
6. **Repeat with `torch.compile(model)`** - if gradients break, flag that we need to disable compile

If compile breaks hooks: override `_do_i_compile()` to return `False` in our trainer.

### Step 1: Copy hyperbolic modules (5 files)

Copy from HyperbodyV1, fix relative imports:
- `lorentz_ops.py` - no changes needed
- `projection_head.py` - `from models.hyperbolic.lorentz_ops` -> `from .lorentz_ops`. **Add `.float()` cast before `exp_map0`**: the 1x1 Conv3d runs in FP16 under AMP autocast, but `exp_map0` needs FP32 to avoid hyperbolic overflow. Insert `x = x.float()` before the `exp_map0` call (the loss already forces FP32 internally, but the projection head is outside the loss scope).
- `label_embedding.py` - same import fix
- `lorentz_loss.py` - same import fix (already has `torch.autocast(enabled=False)` for FP32)
- `organ_hierarchy.py` - no changes needed

Create `__init__.py` re-exporting all public symbols.

**Test:** `test_lorentz_ops.py` - verify exp_map0 shape/inverse, dist functions, NaN-free

### Step 2: Implement `HyperBodyNet` wrapper

**File:** `hyperbolic_network.py`

```python
class HyperBodyNet(nn.Module):
    def __init__(self, backbone, num_classes=70, embed_dim=None, curv=1.0,
                 class_depths=None, min_radius=0.1, max_radius=2.0,
                 direction_mode="random", text_embedding_path=None):
        self.backbone = backbone
        self.hyperbolic_mode = True  # False during inference

        # Dynamically read decoder output channels — no hardcoded 32
        # StackedConvBlocks exposes self.output_channels directly
        in_channels = backbone.decoder.stages[-1].output_channels
        if embed_dim is None:
            embed_dim = in_channels  # default: match decoder output channels

        self.hyp_head = LorentzProjectionHead(in_channels, embed_dim, curv)
        self.label_emb = LorentzLabelEmbedding(num_classes, embed_dim, curv, ...)
        # Hook on last decoder stage
        self._decoder_features = None
        self._hook = backbone.decoder.stages[-1].register_forward_hook(self._capture)

    def _capture(self, module, input, output):
        self._decoder_features = output

    @property
    def decoder(self):  # for set_deep_supervision_enabled
        return self.backbone.decoder

    @property
    def encoder(self):
        return self.backbone.encoder

    def forward(self, x):
        seg_output = self.backbone(x)
        if self.hyperbolic_mode:
            voxel_emb = self.hyp_head(self._decoder_features)
            label_emb = self.label_emb()
            self._decoder_features = None
            return seg_output, voxel_emb, label_emb
        else:
            self._decoder_features = None
            return seg_output

    def compute_conv_feature_map_size(self, input_size):
        return self.backbone.compute_conv_feature_map_size(input_size)
```

**Test:** `test_hyperbolic_network.py`
- Training mode: returns `(seg_output_list, voxel_emb(B,32,D,H,W), label_emb(70,32))`
- Inference mode (`hyperbolic_mode=False`): returns single tensor `(B,70,D,H,W)`
- state_dict includes all backbone + hyperbolic params
- `compute_conv_feature_map_size` delegates correctly

### Step 3: Implement `nnUNetTrainerHyperBody`

**File:** `nnUNetTrainerHyperBody.py`

Override these methods from `nnUNetTrainer`:

| Method | What Changes |
|--------|-------------|
| `__init__` | Add `self.hyp_*` hyperparameters (embed_dim=32, curv=1.0, margin=0.4, weight=0.05, freeze_epochs=5, warmup_epochs=6, etc.) |
| `initialize` | Build backbone via `get_network_from_plans()`, load hierarchy and graph distance (paths resolved via `nnUNet_raw` + dataset name, see Data Dependencies), wrap in `HyperBodyNet`, build hyp_loss (`LorentzTreeRankingLoss` with graph distances). DDP: `find_unused_parameters=True` (needed because `perform_actual_validation()` disables `hyperbolic_mode`, leaving `hyp_head`/`label_emb` unused). |
| `build_network_architecture` | **Static override.** Build PlainConvUNet, wrap in `HyperBodyNet`. Pass `class_depths=None` — safe because at inference `hyperbolic_mode=False` so `label_emb` is never called; weights come from checkpoint. When `enable_deep_supervision=False` (inference), set `hyperbolic_mode=False`. Read `embed_dim` from `arch_init_kwargs["features_per_stage"][0]` instead of hardcoding 32. Critical for `nnUNetPredictor.initialize_from_trained_model_folder` (line 104). |
| `_build_loss` | Same as parent (DC_and_CE + DeepSupervision). Hyp loss built separately. |
| `configure_optimizers` | Build **three optimizers + three schedulers**, store the two hyperbolic pairs as side-effect attributes (`self.optimizer_hyp_head`, `self.optimizer_hyp_emb`, `self.lr_scheduler_hyp_head`, `self.lr_scheduler_hyp_emb`). **Return only** `(optimizer_backbone, lr_scheduler_backbone)` so the base class unpacking `self.optimizer, self.lr_scheduler = self.configure_optimizers()` works. Separate optimizers for hyp_head and label_emb because `PolyLRScheduler` overwrites ALL param_group LRs with the same value — a single multi-group optimizer would destroy the lr ratio. (1) SGD(backbone, lr=0.01) + PolyLR, (2) AdamW(hyp_head, lr=1e-3) + PolyLR, (3) AdamW(label_emb, lr=1e-5) + PolyLR. |
| `train_step` | Forward -> `(seg_output, voxel_emb, label_emb)`. `total_loss = seg_loss + 0.05 * hyp_loss`. Handle target: `target[0].squeeze(1).long()` for hyp_loss. Both optimizers zero_grad / step. Extra grad clip for label_emb on first unfreeze epoch. See "Dual optimizer in train_step" for full AMP pattern. |
| `validation_step` | Same forward as train_step but no backward. Compute both seg_loss and hyp_loss. Return `{'loss': ..., 'hyp_loss': ..., 'tp_hard': ..., 'fp_hard': ..., 'fn_hard': ...}`. Standard Dice metrics from seg_output. |
| `on_train_epoch_start` | **Step all three schedulers:** `self.lr_scheduler.step(epoch)`, `self.lr_scheduler_hyp_head.step(epoch)`, `self.lr_scheduler_hyp_emb.step(epoch)`. Log all LRs. Call `hyp_loss.set_epoch(epoch, max_epochs)` for curriculum. Freeze/unfreeze `label_emb.tangent_embeddings` based on `hyp_freeze_epochs`. |
| `on_train_epoch_end` | Log seg_loss and hyp_loss separately via `print_to_log_file`. |
| `on_validation_epoch_end` | Override to additionally aggregate and log validation `hyp_loss` via `self.logger.log('val_hyp_losses', ...)` and `print_to_log_file`. Call `super()` for standard seg metrics. |
| `set_deep_supervision_enabled` | Toggle both `backbone.decoder.deep_supervision` and `hyperbolic_mode`. |
| `save_checkpoint` | Override to save all three optimizers: `checkpoint['optimizer_state']` for backbone (compatible with base), `checkpoint['optimizer_hyp_head_state']` and `checkpoint['optimizer_hyp_emb_state']` for hyperbolic, plus corresponding `lr_scheduler_hyp_*_state`. |
| `load_checkpoint` | Override entire method (single `torch.load`, no `super()` call) to avoid double-reading large checkpoint files. Load backbone optimizer, hyp_head optimizer, hyp_emb optimizer, and model weights from one dict. Handle missing hyp keys gracefully for backward compat with base nnUNet checkpoints. |
| `_do_i_compile` | Return `False` if hook gradient test fails under compile (determined in Step 0). |

### Step 4: Optional pretrained backbone loading

Add `load_pretrained_backbone(checkpoint_path)` utility method (not used by default since training from scratch):
- Load nnUNet checkpoint, remap keys with `backbone.` prefix
- `load_state_dict(remapped, strict=False)`

### Step 5: Integration tests

**Test:** `test_integration.py`
- One `train_step` with synthetic data: loss is scalar, backward succeeds
- AMP compatibility: no NaN with GradScaler
- **AMP dual-optimizer inf/nan recovery:** Force an inf gradient (e.g., scale a param to 1e38), verify both optimizers skip their `step()`, `grad_scaler` reduces scale, next step recovers normally
- Gradient flow: gradients reach backbone (SGD), hyp_head (AdamW), label_emb (when unfrozen)
- Both optimizers update their respective params
- **Checkpoint round-trip:** `save_checkpoint` → `load_checkpoint`, verify both optimizer states and LR scheduler states match

---

## Key Technical Details

### configure_optimizers pattern
```python
def configure_optimizers(self):
    # Backbone: SGD (proven for nnUNet)
    optimizer_backbone = SGD(self.network.backbone.parameters(),
                             self.initial_lr, weight_decay=self.weight_decay,
                             momentum=0.99, nesterov=True)
    lr_scheduler_backbone = PolyLRScheduler(optimizer_backbone, self.initial_lr, self.num_epochs)

    # Hyperbolic: AdamW with differential LR
    # WARNING: PolyLRScheduler overwrites ALL param_group LRs with the same value
    # (see polylr.py:18-20). A single scheduler on a multi-group optimizer would
    # destroy the LR ratio (label_emb lr jumps from 1e-5 to 1e-3).
    # Fix: two separate single-group optimizers, each with its own scheduler.
    self.optimizer_hyp_head = AdamW(self.network.hyp_head.parameters(), lr=self.hyp_lr)
    self.optimizer_hyp_emb = AdamW(self.network.label_emb.parameters(),
                                    lr=self.hyp_lr * self.hyp_text_lr_ratio)
    self.lr_scheduler_hyp_head = PolyLRScheduler(self.optimizer_hyp_head, self.hyp_lr, self.num_epochs)
    self.lr_scheduler_hyp_emb = PolyLRScheduler(self.optimizer_hyp_emb,
                                                 self.hyp_lr * self.hyp_text_lr_ratio, self.num_epochs)

    # Return only backbone pair — base class assigns to self.optimizer / self.lr_scheduler
    return optimizer_backbone, lr_scheduler_backbone
```

### Dual optimizer in train_step
```python
self.optimizer.zero_grad(set_to_none=True)       # backbone (= self.optimizer from base)
self.optimizer_hyp_head.zero_grad(set_to_none=True)
self.optimizer_hyp_emb.zero_grad(set_to_none=True)

with autocast(...):
    seg_output, voxel_emb, label_emb = self.network(data)
    seg_loss = self.loss(seg_output, target)
    hyp_loss = self.hyp_loss(voxel_emb, target[0].squeeze(1).long(), label_emb)
    total_loss = seg_loss + self.hyp_weight * hyp_loss

# Single backward, three optimizer steps
# NOTE: if unscale_ detects inf/nan, ALL optimizers skip their step and
# grad_scaler reduces the scale factor. This is correct PyTorch behavior —
# a single loss backward produces one set of scaled gradients shared by all.
if self.grad_scaler is not None:
    self.grad_scaler.scale(total_loss).backward()
    self.grad_scaler.unscale_(self.optimizer)
    self.grad_scaler.unscale_(self.optimizer_hyp_head)
    self.grad_scaler.unscale_(self.optimizer_hyp_emb)
    clip_grad_norm_(self.network.backbone.parameters(), 12)
    if self._is_unfreeze_epoch():
        clip_grad_norm_(self.network.label_emb.parameters(), self.hyp_text_grad_clip)
    self.grad_scaler.step(self.optimizer)
    self.grad_scaler.step(self.optimizer_hyp_head)
    self.grad_scaler.step(self.optimizer_hyp_emb)
    self.grad_scaler.update()   # called ONCE after all optimizer steps
else:
    total_loss.backward()
    clip_grad_norm_(self.network.backbone.parameters(), 12)
    if self._is_unfreeze_epoch():
        clip_grad_norm_(self.network.label_emb.parameters(), self.hyp_text_grad_clip)
    self.optimizer.step()
    self.optimizer_hyp_head.step()
    self.optimizer_hyp_emb.step()
```

### Checkpoint save/load
```python
def save_checkpoint(self, filename):
    if self.local_rank == 0 and not self.disable_checkpointing:
        # ... (same as base class checkpoint dict construction) ...
        checkpoint['optimizer_hyp_head_state'] = self.optimizer_hyp_head.state_dict()
        checkpoint['optimizer_hyp_emb_state'] = self.optimizer_hyp_emb.state_dict()
        checkpoint['lr_scheduler_hyp_head_state'] = self.lr_scheduler_hyp_head.state_dict()
        checkpoint['lr_scheduler_hyp_emb_state'] = self.lr_scheduler_hyp_emb.state_dict()
        torch.save(checkpoint, filename)

def load_checkpoint(self, filename_or_checkpoint):
    # Override entire method — do NOT call super().load_checkpoint().
    # Reason: super() calls torch.load internally, then we'd need to call it
    # again for hyp keys. For a 3D UNet checkpoint (hundreds of MB) this
    # doubles memory peak and I/O. Instead, load once and handle all keys.
    if not self.was_initialized:
        self.initialize()
    if isinstance(filename_or_checkpoint, str):
        checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
    else:
        checkpoint = filename_or_checkpoint

    # --- model weights (same logic as base class) ---
    # TODO: copy lines 1182-1206 from nnUNetTrainer.load_checkpoint verbatim.
    # This includes: module. prefix stripping, DDP unwrap (self.network.module),
    # OptimizedModule unwrap (._orig_mod), and load_state_dict(strict=True).
    # Do NOT simplify — edge cases matter (e.g. checkpoint from DDP loaded into
    # non-DDP, or vice versa).
    new_state_dict = {}
    for k, value in checkpoint['network_weights'].items():
        key = k
        if key not in self.network.state_dict().keys() and key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    if self.is_ddp:
        if isinstance(self.network.module, OptimizedModule):
            self.network.module._orig_mod.load_state_dict(new_state_dict)
        else:
            self.network.module.load_state_dict(new_state_dict)
    else:
        if isinstance(self.network, OptimizedModule):
            self.network._orig_mod.load_state_dict(new_state_dict)
        else:
            self.network.load_state_dict(new_state_dict)

    # --- base trainer state ---
    self.my_init_kwargs = checkpoint['init_args']
    self.current_epoch = checkpoint['current_epoch']
    self.logger.load_checkpoint(checkpoint['logging'])
    self._best_ema = checkpoint['_best_ema']
    self.inference_allowed_mirroring_axes = checkpoint.get(
        'inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes)
    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
    if self.grad_scaler is not None and checkpoint.get('grad_scaler_state') is not None:
        self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    # --- hyperbolic state (graceful if missing — enables loading base nnUNet checkpoints) ---
    if 'optimizer_hyp_head_state' in checkpoint:
        self.optimizer_hyp_head.load_state_dict(checkpoint['optimizer_hyp_head_state'])
    if 'optimizer_hyp_emb_state' in checkpoint:
        self.optimizer_hyp_emb.load_state_dict(checkpoint['optimizer_hyp_emb_state'])
    if 'lr_scheduler_hyp_head_state' in checkpoint:
        self.lr_scheduler_hyp_head.load_state_dict(checkpoint['lr_scheduler_hyp_head_state'])
    if 'lr_scheduler_hyp_emb_state' in checkpoint:
        self.lr_scheduler_hyp_emb.load_state_dict(checkpoint['lr_scheduler_hyp_emb_state'])
```

### on_train_epoch_start (triple scheduler stepping)
```python
def on_train_epoch_start(self):
    self.network.train()
    # Intentionally NOT calling super() — we need to control all three schedulers
    # and the base class only handles self.lr_scheduler. If nnUNet updates
    # on_train_epoch_start in the future, review this override.
    self.lr_scheduler.step(self.current_epoch)
    self.lr_scheduler_hyp_head.step(self.current_epoch)
    self.lr_scheduler_hyp_emb.step(self.current_epoch)
    self.print_to_log_file('')
    self.print_to_log_file(f'Epoch {self.current_epoch}')
    self.print_to_log_file(
        f"Backbone LR: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}, "
        f"HypHead LR: {np.round(self.optimizer_hyp_head.param_groups[0]['lr'], decimals=5)}, "
        f"LabelEmb LR: {np.round(self.optimizer_hyp_emb.param_groups[0]['lr'], decimals=5)}"
    )
    # Curriculum temperature scheduling
    self.hyp_loss.set_epoch(self.current_epoch, self.num_epochs)
    # Freeze/unfreeze label embeddings
    mod = self.network.module if self.is_ddp else self.network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod
    if self.current_epoch < self.hyp_freeze_epochs:
        mod.label_emb.tangent_embeddings.requires_grad_(False)
    else:
        mod.label_emb.tangent_embeddings.requires_grad_(True)
```

### Target format conversion
nnUnet targets: `(B, 1, D, H, W)` float (or list with deep supervision where `target[0]` is full-res). LorentzRankingLoss expects `(B, D, H, W)` int64. Conversion: `target[0].squeeze(1).long()`.

### on_validation_epoch_end (hyp_loss logging)
```python
def on_validation_epoch_end(self, val_outputs):
    # Extract hyp_loss before calling super (which only uses loss/tp/fp/fn)
    outputs_collated = collate_outputs(val_outputs)
    hyp_loss_here = np.mean(outputs_collated['hyp_loss'])
    if self.is_ddp:
        hyp_losses_val = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(hyp_losses_val, outputs_collated['hyp_loss'])
        hyp_loss_here = np.vstack(hyp_losses_val).mean()
    self.logger.log('val_hyp_losses', hyp_loss_here, self.current_epoch)
    # Delegate standard metrics (Dice, seg_loss) to parent
    super().on_validation_epoch_end(val_outputs)
```

### Inference compatibility
`nnUNetPredictor.initialize_from_trained_model_folder` (line 104 of `predict_from_raw_data.py`) calls `trainer_class.build_network_architecture(enable_deep_supervision=False)`. Our override returns `HyperBodyNet` with `hyperbolic_mode=False` when `deep_supervision=False`, so `forward()` returns only seg logits. Inference pipeline works unchanged.

**Caveat:** The static method passes `class_depths=None` to `LorentzLabelEmbedding`, which means the label embeddings are initialized with default (uniform) tangent norms instead of hierarchy-aware norms. This is safe because: (a) `hyperbolic_mode=False` means `label_emb()` is never called during inference, and (b) the actual trained weights are loaded from the checkpoint immediately after construction. Do NOT call `forward()` in hyperbolic mode on a model built this way without first loading checkpoint weights.

### `torch.compile` strategy
Test hooks under compile in Step 0. If broken: override `_do_i_compile()` → `return False`. This is acceptable since the hyperbolic branch adds minimal compute vs the 3D convolutions.

---

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| hyp_embed_dim | 32 | HyperbodyV1 default |
| hyp_curv | 1.0 | HyperbodyV1 default |
| hyp_margin | 0.4 | HyperbodyV1 default |
| hyp_samples_per_class | 64 | HyperbodyV1 default |
| hyp_num_negatives | 8 | HyperbodyV1 default |
| hyp_weight | 0.05 | HyperbodyV1 default |
| hyp_t_start | 2.0 | HyperbodyV1 default |
| hyp_t_end | 0.1 | HyperbodyV1 default |
| hyp_warmup_epochs | 6 | HyperbodyV1 default |
| hyp_freeze_epochs | 5 | HyperbodyV1 default |
| hyp_text_lr_ratio | 0.01 | HyperbodyV1 default |
| hyp_text_grad_clip | 0.1 | HyperbodyV1 default |
| hyp_distance_mode | "graph" | User choice |
| hyp_direction_mode | "random" | User choice |
| backbone_lr | 0.01 | nnUnet default |
| hyp_lr | 1e-3 | HyperbodyV1 default |
| label_emb_lr | 1e-5 | 1e-3 * 0.01 ratio |

---

## Data Dependencies

| File | Path | Purpose |
|------|------|---------|
| tree.json | `nnUNet_raw/{dataset_name}/tree.json` — resolved at runtime via `join(nnUNet_raw, self.plans_manager.dataset_name, "tree.json")` | Organ hierarchy for class_depths |
| graph_distance_matrix.pt | `nnUNet_raw/{dataset_name}/graph_distance_matrix.pt` — same path resolution | Precomputed graph distances for LorentzTreeRankingLoss |
| nnUNet plans | `nnUNet_preprocessed/Dataset501_HyperBody/nnUNetPlans.json` (available as `self.plans_manager`) | Network architecture config (includes `features_per_stage`) |
| dataset.json | `nnUNet_raw/Dataset501_HyperBody/dataset.json` | Class labels (70 classes) |

---

## Verification Plan

1. **Step 0 gate:** Hook gradient flow test passes (with/without compile)
2. **Unit tests pass:** All 5 test files green
3. **1-epoch smoke test:**
   ```bash
   conda activate nnunet
   nnUNetv2_train Dataset501_HyperBody 3d_fullres 0 -tr nnUNetTrainerHyperBody --npz
   ```
   Verify: log shows seg_loss + hyp_loss, no NaN, checkpoint saved
4. **Inference:** Run `perform_actual_validation()`, verify valid segmentation masks
5. **Full training:** Compare Dice with baseline nnUNet

---

## Critical Source Files (Reference)

| File | Key Content |
|------|------------|
| `nnUnet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:200-238` | `initialize()` |
| `nnUnet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:304-336` | `build_network_architecture()` |
| `nnUnet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:507-511` | `configure_optimizers()` |
| `nnUnet/nnunetv2/training/lr_scheduler/polylr.py` | `PolyLRScheduler` — overwrites ALL param_group LRs with same value (line 18-20) |
| `nnUnet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:973-1003` | `train_step()` |
| `nnUnet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:880-892` | `set_deep_supervision_enabled()` |
| `nnUnet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1149-1210` | `save_checkpoint()` / `load_checkpoint()` |
| `nnUnet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1080-1118` | `on_validation_epoch_end()` |
| `nnUnet/nnunetv2/inference/predict_from_raw_data.py:67-118` | Inference loading path |
| `~/.../dynamic_network_architectures/.../unet_decoder.py:100-125` | `UNetDecoder.forward()` - hook target |
| `HyperbodyV1/models/body_net.py:17-114` | `BodyNet` reference wrapper |
| `HyperbodyV1/train.py:139-221` | Training loop reference |
| `HyperbodyV1/train.py:452-460` | Optimizer param groups reference |
| `HyperbodyV1/train.py:536-549` | Curriculum + freeze/unfreeze reference |
| `HyperbodyV1/data/organ_hierarchy.py` | Hierarchy parsing (to copy) |

---

## Revision Log

### Rev 1 — Post-Review Fixes

Issues found during code review against nnUNet base class source. Changes applied:

**Critical (runtime blockers):**

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | `save_checkpoint()` / `load_checkpoint()` crash with dual optimizer — base class saves `self.optimizer` (singular) | Added `save_checkpoint` and `load_checkpoint` overrides to Step 3 method table + Key Technical Details code snippet. `load_checkpoint` overrides entire method (no `super()`) to avoid double `torch.load` on large checkpoints (Issue A). Saves all hyp optimizer/scheduler states. Graceful fallback when loading base nnUNet checkpoints. |
| 2 | `configure_optimizers()` return breaks `self.optimizer, self.lr_scheduler = ...` unpacking in `initialize()` | Changed to side-effect pattern: store `self.optimizer_hyp_head` / `self.optimizer_hyp_emb` / `self.lr_scheduler_hyp_head` / `self.lr_scheduler_hyp_emb` as attributes inside `configure_optimizers()`, return only backbone pair. Added full code snippet. |
| 3 | `on_train_epoch_start()` only steps one LR scheduler — hyp schedulers would never update | Added override that steps all three schedulers, logs all LRs, and includes curriculum + freeze/unfreeze logic. Intentionally not calling `super()` — documented why (Issue B). Full code snippet added. |

**Important (logic gaps / silent bugs):**

| # | Issue | Fix Applied |
|---|-------|-------------|
| 4 | `build_network_architecture` static method cannot access hierarchy data | Updated method description: pass `class_depths=None` at inference, documented why this is safe (weights from checkpoint, hyperbolic_mode=False). Added caveat to Inference Compatibility section. |
| 5 | Data file paths (`Dataset/tree.json`) unresolvable | Fixed paths to use `nnUNet_raw/{dataset_name}/` resolution via `self.plans_manager.dataset_name`. Updated Data Dependencies table. |
| 7 | AMP dual-optimizer: if `unscale_` detects inf/nan, both optimizers skip | Added explanatory comment to train_step code snippet. Added explicit AMP inf/nan recovery test to Step 5 integration tests. |
| 8 | Validation `hyp_loss` computed but never logged | Added `on_validation_epoch_end` override to method table. Added full code snippet with DDP-aware aggregation. |
| 10 | `on_validation_epoch_end` missing from override list | Now listed in Step 3 method table with description. |
| 12 | AMP FP16 in projection head 1x1 conv before `exp_map0` | Added `.float()` cast instruction to Step 1 `projection_head.py` copy notes. |

### Rev 2 — PolyLRScheduler Bug & Cleanup

| # | Issue | Fix Applied |
|---|-------|-------------|
| C | **`PolyLRScheduler` destroys per-group LR ratio** — `polylr.py:18-20` computes one `new_lr` and overwrites ALL param groups. A single multi-group AdamW for hyp_head(1e-3) + label_emb(1e-5) would set both to 1e-3 at first step, 100x blowup on label_emb. | Split into two separate single-group optimizers: `optimizer_hyp_head` (AdamW, lr=1e-3) and `optimizer_hyp_emb` (AdamW, lr=1e-5), each with its own `PolyLRScheduler`. Updated all code snippets (configure_optimizers, train_step, checkpoint, on_train_epoch_start) and method table. |
| A | `load_checkpoint` double `torch.load` — `super()` loads once, then we load again for hyp keys. Doubles memory peak for large 3D UNet checkpoints. | Changed `load_checkpoint` to override entire method: single `torch.load`, handle all keys (model weights + backbone optimizer + hyp optimizers) in one pass. No `super()` call. |
| B | `on_train_epoch_start` doesn't call `super()` — may miss future nnUNet updates. | Added explicit comment: "intentionally not calling super() because we need to control all three schedulers". |
| 6 | `HyperBodyNet.__init__` had undefined `in_channels` and hardcoded `embed_dim=32` | Fixed: read `in_channels = backbone.decoder.stages[-1].output_channels` (StackedConvBlocks exposes this directly). Default `embed_dim=None` → falls back to `in_channels`. |
