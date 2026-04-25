## SiamAPN++ + MobileOne-S2 - Implementation Plan

### What you're building

SiamAPN++ with its AlexNet backbone replaced by MobileOne-S2, while preserving the real upstream SiamAPN++ architecture as closely as possible.

After inspecting the real upstream code, these points are now confirmed:

- the correct upstream repository is `https://github.com/vision4robotics/SiamAPN`
- the `SiamAPN++` implementation lives inside that repo under `SiamAPN++/`
- the real backbone contract is a 2-tuple consumed directly by the upstream `APN` and `clsandloc` modules
- the first correct integration target is to preserve that contract, not redesign the head

The safest faithful integration path is:

1. clone and inspect the upstream SiamAPN++ implementation
2. replace AlexNet with MobileOne-S2 inside the upstream backbone slot
3. adapt MobileOne features back to the channels expected by upstream SiamAPN++
4. only then adapt training data to the competition dataset

---

### Step 1 - Environment & Repos

```bash
git clone https://github.com/vision4robotics/SiamAPN
git clone https://github.com/apple/ml-mobileone

cd SiamAPN/SiamAPN++
pip install -r ../requirement.txt
```

Download MobileOne-S2 ImageNet weights from the Apple repo:
`mobileone_s2_unfused.pth.tar` - the unfused version is required for training. The fused version is inference-only.

---

### Step 2 - Confirm the Upstream SiamAPN++ Contract

From the real upstream implementation:

- `ModelBuilderADAPN` uses `self.backbone(template)` and `self.backbone(search)`
- the backbone must return two feature maps
- those two feature maps are consumed by:
  - `APN` in stage-1
  - `clsandloc` in stage-2

The actual AlexNet outputs are:

- template `127 x 127` -> `layer4: [B, 384, 8, 8]`, `layer5: [B, 256, 6, 6]`
- search `287 x 287` -> `layer4: [B, 384, 28, 28]`, `layer5: [B, 256, 26, 26]`

This means the backbone swap is not just "return any two stages." It must preserve a tuple contract that the rest of SiamAPN++ already assumes.

---

### Step 3 - Backbone Swap

Replace the upstream AlexNet backbone in `pysot/models/model_builder_adapn.py`:

```python
self.backbone = AlexNet().cuda()
```

with a MobileOne-S2 backbone module that still returns a tuple shaped for SiamAPN++.

The safest first version is:

- take two internal MobileOne-S2 stages
- add lightweight `1x1` adapters
- return:
  - feature A adapted to `384` channels for the APN branch
  - feature B adapted to `256` channels for the `clsandloc` branch

That lets the existing `APN` and `clsandloc` modules remain unchanged in the first pass.

Example shape intent:

```python
class MobileOneS2Backbone(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.model = mobileone(variant="s2")
        self.adapter_apn = nn.Conv2d(..., 384, kernel_size=1)
        self.adapter_head = nn.Conv2d(..., 256, kernel_size=1)

        if pretrained:
            state = torch.load(pretrained)["state_dict"]
            self.model.load_state_dict(state, strict=False)

    def forward(self, x):
        ...
        feat_apn = self.adapter_apn(stage_feature_a)
        feat_head = self.adapter_head(stage_feature_b)
        return feat_apn, feat_head
```

Important correction:

- do not start by rewriting `APN` input channels to native MobileOne channels
- first make the new backbone satisfy the old contract

---

### Step 4 - Shape Verification Before Further Edits

Before touching losses, tracker logic, or dataset code, verify that:

- template input is still `127 x 127`
- search input is still `287 x 287`
- the new backbone returns two tensors compatible with all existing depthwise correlations
- the final response maps still align with upstream `OUTPUT_SIZE = 21`

This should be checked with a forward-shape test on:

- backbone only
- full `ModelBuilderADAPN`

The first milestone is:
"MobileOne backbone swapped in, upstream SiamAPN++ forward pass still works."

---

### Step 5 - Inference Reparameterization

This is the step people forget. MobileOne at training time has multiple parallel branches per block. At inference you must fuse them or you lose the speed benefit:

```python
from mobileone import reparameterize_model

# After loading checkpoint, before any inference:
tracker.backbone.model = reparameterize_model(tracker.backbone.model)
```

Add this in the tracker initialization path so it happens once per sequence, not per frame.

---

### Step 6 - Training Config

Start from the real upstream `SiamAPN++/experiments/config.yaml` instead of introducing a separate config shape too early.

Important corrections from the upstream code:

- training search size is `287`, not `255`
- output size is `21`
- backbone train layers are currently AlexNet layers and must be redefined for MobileOne

The first faithful training config should keep the original SiamAPN++ geometry:

```yaml
BACKBONE:
  TYPE: MobileOneS2
  PRETRAINED: 'mobileone_s2_unfused.pth.tar'
  TRAIN_LAYERS: ['stage2', 'stage3']
  TRAIN_EPOCH: 10
  LAYERS_LR: 0.1

TRAIN:
  EPOCH: 50
  SEARCH_SIZE: 287
  OUTPUT_SIZE: 21
  BASE_LR: 0.005
  BATCH_SIZE: 32
  NUM_WORKERS: 4
```

Freeze early MobileOne stages first, then unfreeze the selected train layers using the same staged-training strategy already used in upstream SiamAPN++.

---

### Step 7 - Dataset Adaptation

Only after the backbone swap is shape-stable should the dataset be adapted to the competition data.

This is another important correction to the original plan:

- the first task is not "build a fresh generic dataloader"
- the faithful task is to adapt the competition data into the exact sample structure expected by upstream SiamAPN++

The upstream training pipeline expects a sample dict containing:

- `template`
- `search`
- `bbox`
- `label_cls2`
- `labelxff`
- `labelcls3`
- `weightxff`

So the competition data adapter must:

1. sample valid template/search frame pairs from the competition train split
2. generate SiamFC-style crops
3. produce the extra APN++ supervision targets expected by `AnchorTarget`

Only then should training be run on the competition data.

---

### Step 8 - Verify Budget Before Training

Before spending GPU hours, confirm the final integrated tracker stays under the competition limits:

```python
from fvcore.nn import FlopCountAnalysis, parameter_count
import torch

model.eval()
template = torch.randn(1, 3, 127, 127)
search = torch.randn(1, 3, 287, 287)

flops = FlopCountAnalysis(model, (template, search))
params = parameter_count(model)

print(f"GFLOPs: {flops.total() / 1e9:.1f}")   # target: < 30
print(f"Params: {params[''] / 1e6:.1f}M")     # target: < 50M
```

Measure this on the real integrated SiamAPN++ + MobileOne-S2 model, not on a simplified proxy.

---

### Step 9 - Migrate the Finalized Upstream-Based Code Into This Repo

Once the upstream-based implementation is stable and verified, migrate only the code that is actually needed into this repository so the final project is clean and self-contained.

The migration should happen after the architecture is validated, not before, so that:

- only the real required files are copied
- unnecessary upstream utilities are not dragged into the repo
- imports and config paths can be rewritten once instead of repeatedly

The migration target should include:

- the finalized SiamAPN++ model code actually in use
- the finalized MobileOne-S2 backbone wrapper
- the competition-adapted SiamAPN++ dataset code
- the required anchor-target and augmentation helpers
- the local training entry point and config files needed for this repo

The migration should avoid copying large unused upstream areas such as:

- old public dataset preprocessing scripts not needed for competition training
- evaluation/toolkit code not used by this project
- unused demo/test scripts
- unused backbone or tracker variants

The active port now runs from this repository without depending on `external/SiamAPN` at runtime. Source mapping and notices are preserved in `docs/references/siamapn_port_provenance.md` and `docs/references/third_party_notices.md`.

---

### Step 10 - Remove Temporary External Code

The migrated code is active in this repository:

1. runtime imports resolve to active `models/`, `data/`, `app/`, and `evaluation/` modules
2. provenance and third-party notices are preserved under `docs/references/`
3. parity/golden tests cover target generation, losses, tracker decode helpers, and MobileOne feature/reparameterization contracts
4. the temporary `external/` reference copy can be deleted

The final project structure does not require `external/`.

---

### What "meaningful contribution" looks like here

The competition requires participant contribution beyond off-the-shelf models. In this corrected plan, your contribution is:

1. MobileOne-S2 backbone replacing AlexNet inside the real SiamAPN++ architecture
2. re-parameterization at inference for efficiency
3. adaptation of SiamAPN++ training to the competition domain data

That is both defensible and clearly stronger than using an off-the-shelf tracker unchanged.
