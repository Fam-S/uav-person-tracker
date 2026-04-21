## SiamAPN++ + MobileOne-S2 — Implementation Plan

### What you're building

SiamAPN++ with its AlexNet backbone replaced by MobileOne-S2. The tracker's dual-level feature fusion (which replaces AlexNet's last two conv layers) maps naturally onto MobileOne's stage 3 and stage 4 outputs — same idea, better features.

---

### Step 1 — Environment & Repos

```bash
git clone https://github.com/vision4robotics/SiamAPN-plusplus
git clone https://github.com/apple/ml-mobileone

cd SiamAPN-plusplus
pip install -r requirements.txt
```

Download MobileOne-S2 ImageNet weights from the Apple repo:
`mobileone_s2_unfused.pth.tar` — the **unfused** version is required for training. The fused version is inference-only.

---

### Step 2 — Backbone Swap

SiamAPN++ uses AlexNet's `conv4` and `conv5` outputs as its two feature levels. MobileOne-S2 has 4 stages — you'll tap **stage 3** (semantic, coarser) and **stage 2** (spatial, finer) as the direct replacement.

```python
# In backbone/mobileone.py — expose two outputs
class MobileOneS2Backbone(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.model = mobileone(variant='s2')
        if pretrained:
            state = torch.load(pretrained)['state_dict']
            self.model.load_state_dict(state, strict=False)
            # strict=False drops the ImageNet classifier head silently

    def forward(self, x):
        # MobileOne stages: stem → stage0 → stage1 → stage2 → stage3 → stage4
        x = self.model.stem(x)
        x = self.model.stages[0](x)
        x = self.model.stages[1](x)
        f1 = self.model.stages[2](x)   # → replaces AlexNet conv4
        f2 = self.model.stages[3](f1)  # → replaces AlexNet conv5
        return f1, f2                   # dual-level output, same contract as AlexNet
```

Then in `siamapn.py`, replace `self.backbone = AlexNet(...)` with `self.backbone = MobileOneS2Backbone(pretrained=cfg.BACKBONE.PRETRAINED)`.

The channel dimensions will be different from AlexNet — you'll need to adjust the **APN neck's input channels** to match MobileOne-S2's stage 2/3 output channels (256 and 512 respectively vs AlexNet's 256/256). This is a one-line change in the neck's `__init__`.

---

### Step 3 — Inference Reparameterization

This is the step people forget. MobileOne at training time has multiple parallel branches per block. At inference you must fuse them or you lose all the speed:

```python
from mobileone import reparameterize_model

# After loading checkpoint, before any inference:
tracker.backbone.model = reparameterize_model(tracker.backbone.model)
```

Add this to the tracker's `init()` method so it happens once per sequence, not per frame.

---

### Step 4 — Training Config

In `experiments/siamapnpp_mobileone.yaml`:

```yaml
BACKBONE:
  TYPE: MobileOneS2
  PRETRAINED: 'pretrained/mobileone_s2_unfused.pth.tar'
  TRAIN_LAYERS: ['stages.2', 'stages.3']  # only fine-tune last two stages
  TRAIN_LR: 1e-5   # 10x lower than head

TRAIN:
  EPOCHS: 30
  LR: 1e-4          # head learning rate
  LR_SCHEDULER: cosine
  BATCH_SIZE: 32    # fits Colab T4
  NUM_WORKERS: 4

DATASET:
  TRAIN:
    - GOT10K
    - COCO
    - competition_train   # your 255 sequences
  # DO NOT add UAV123, DTB70, UAVTrack112, UAV20L — contaminated
```

Freeze stages 0 and 1 entirely for the first 10 epochs, then unfreeze stage 2 with the lower LR. This prevents early training from destroying the low-level features before the head stabilizes.

---

### Step 5 — Channel Mismatch Fix (the likely first error)

When you first run, you'll almost certainly get a shape mismatch in the APN neck. MobileOne-S2 outputs:

| Stage | Channels | Stride |
|---|---|---|
| Stage 2 | 256 | 8× |
| Stage 3 | 512 | 16× |

AlexNet outputs 256 channels at both levels. So in `neck/apn.py`, find the `in_channels` arguments and update them:

```python
# Before (AlexNet):
self.apn = APN(in_channels=[256, 256], ...)

# After (MobileOne-S2):
self.apn = APN(in_channels=[256, 512], ...)
```

---

### Step 6 — Verify Budget Before Training

Before spending GPU hours, confirm you're under the competition limits:

```python
from fvcore.nn import FlopCountAnalysis, parameter_count
import torch

model.eval()
template = torch.randn(1, 3, 127, 127)
search   = torch.randn(1, 3, 255, 255)

flops = FlopCountAnalysis(model, (template, search))
params = parameter_count(model)

print(f"GFLOPs: {flops.total() / 1e9:.1f}")   # target: < 30
print(f"Params: {params[''] / 1e6:.1f}M")      # target: < 50M
```

Expected: ~3–5 GFLOPs, ~12–15M params — well inside budget.

---

### What "meaningful contribution" looks like here

The competition requires participant contribution beyond off-the-shelf models. Yours is:
1. MobileOne-S2 backbone replacing AlexNet — novel combination, not in any paper
2. Re-parameterization at inference — explicit efficiency contribution
3. Fine-tuned on competition domain data

That's sufficient and defensible.
