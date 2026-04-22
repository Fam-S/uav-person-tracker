## Colab GPU Smoke Test

Verify that the SiamAPN++ + MobileOne-S2 integration works correctly on a Google Colab GPU.

### 1. Start Colab With GPU

1. `Runtime` -> `Change runtime type`
2. Set `Hardware accelerator` to `GPU`

### 2. Clone The Repo

```bash
!git clone https://github.com/Fam-S/uav-person-tracker.git
```

### 3. Verify Working Directory

```bash
!pwd
!ls
```

### 4. Install UV

```bash
!curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 5. Install Dependencies

```bash
!uv sync
```

### 6. Download MobileOne-S2 Unfused Weights

```bash
!mkdir -p external/SiamAPN/SiamAPN++/pretrained_models
!curl -L -o external/SiamAPN/SiamAPN++/pretrained_models/mobileone_s2_unfused.pth.tar https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s2_unfused.pth.tar
```

### 7. Verify CUDA

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
```

### 8. Shape Verification

Before touching losses or training, verify the backbone swap produces correct shapes for the SiamAPN++ contract.

```python
import torch

from models.backbone import MobileOneS2Backbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = MobileOneS2Backbone(pretrained_path=None, normalize_input=True).to(device)

template = torch.rand(1, 3, 127, 127, device=device)
search = torch.rand(1, 3, 287, 287, device=device)

t_low, t_high = backbone(template)
s_low, s_high = backbone(search)

print("template 127x127 -> low:", tuple(t_low.shape), "high:", tuple(t_high.shape))
print("search   287x287 -> low:", tuple(s_low.shape), "high:", tuple(s_high.shape))
```

Both outputs must be 4D tensors (batch, channels, H, W). The two feature levels should have different spatial sizes.

### 9. Full Model Forward Pass

Verify the full SiamAPN++ + MobileOne-S2 model produces valid output on GPU.

```python
import torch

from models import SiamAPNppMobileOne

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiamAPNppMobileOne(feature_channels=96, pretrained_path=None).to(device)
template = torch.rand(1, 3, 127, 127, device=device)
search = torch.rand(1, 3, 255, 255, device=device)

outputs = model(template, search)
print("bbox_pred shape:", tuple(outputs["bbox_pred"].shape))
assert outputs["bbox_pred"].shape == (1, 4), "Expected (1, 4) bbox prediction"
print("Forward pass OK")
```

### 10. Backward Pass & Loss Check

Verify that loss computation and gradient flow work on GPU.

```python
import torch

from models import SiamAPNppMobileOne
from models.losses import SiamAPNLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiamAPNppMobileOne(feature_channels=96, pretrained_path=None).to(device)
criterion = SiamAPNLoss(search_size=255).to(device)

template = torch.rand(1, 3, 127, 127, device=device)
search = torch.rand(1, 3, 255, 255, device=device)
search_bbox = torch.tensor([[60.0, 70.0, 50.0, 80.0]], device=device)

outputs = model(template, search)
loss_out = criterion(bbox_pred=outputs["bbox_pred"], search_bbox=search_bbox)

print("total_loss:", loss_out.total_loss.item())
print("reg_loss:", loss_out.reg_loss.item())
assert torch.isfinite(loss_out.total_loss), "Loss is not finite"

loss_out.total_loss.backward()
grads_exist = any(p.grad is not None for p in model.parameters() if p.requires_grad)
print("gradients flow:", grads_exist)
assert grads_exist, "No gradients flowing"
print("Backward pass OK")
```

### 11. FLOP & Parameter Budget Check

Confirm the model stays under competition limits.

```python
import torch

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("fvcore not installed, skipping budget check. Run: pip install fvcore")

if HAS_FVCORE:
    from models import SiamAPNppMobileOne

    model = SiamAPNppMobileOne(feature_channels=96, pretrained_path=None).cuda()
    model.eval()

    template = torch.randn(1, 3, 127, 127).cuda()
    search = torch.randn(1, 3, 255, 255).cuda()

    flops = FlopCountAnalysis(model, (template, search))
    params = parameter_count(model)

    gflops = flops.total() / 1e9
    mparams = params[""] / 1e6

    print(f"GFLOPs: {gflops:.1f}  (target: < 30)")
    print(f"Params: {mparams:.1f}M  (target: < 50M)")
```

### 12. Run Architecture Tests

```bash
!uv sync --group dev
!uv run pytest tests/test_architecture.py -v
```

### What Success Looks Like

All steps above complete without errors:

- CUDA check prints `cuda available: True`
- Backbone returns two 4D feature maps at different spatial sizes
- Full model forward pass produces `(1, 4)` bbox prediction
- Loss is finite and gradients flow through the model
- FLOPs < 30 GFLOPs and params < 50M
- All architecture tests pass
