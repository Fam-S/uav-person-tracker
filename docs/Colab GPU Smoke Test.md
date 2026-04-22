## Colab GPU Smoke Test

This is the smallest faithful smoke test for the upstream-based `SiamAPN++ + MobileOne-S2` integration.

### Goal

Confirm on a CUDA machine that the real upstream pipeline can:

- build the model
- load MobileOne-S2 pretrained weights
- build the competition dataset
- compute the SiamAPN++ losses
- run backward
- save a checkpoint

### Before You Start

You need:

- this repo available in Colab
- your competition dataset available in Colab or Google Drive
- `mobileone_s2_unfused.pth.tar`

Expected upstream paths:

- config: `external/SiamAPN/SiamAPN++/experiments/config.yaml`
- train script: `external/SiamAPN/SiamAPN++/tools/train_apn++.py`
- pretrained weights folder: `external/SiamAPN/SiamAPN++/pretrained_models/`

### 1. Start Colab With GPU

In Colab:

1. `Runtime` -> `Change runtime type`
2. Set `Hardware accelerator` to `GPU`

### 2. Clone Or Mount The Repo

If cloning from GitHub:

```bash
git clone <YOUR-REPO-URL> /content/uav-person-tracker
cd /content/uav-person-tracker
```

If using Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/uav-person-tracker
```

### 3. Install Python Dependencies With UV

Run:

```bash
cd /content/uav-person-tracker
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --dev
```

Quick CUDA check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

### 4. Put MobileOne Weights In The Expected Folder

The training script expects:

```text
external/SiamAPN/SiamAPN++/pretrained_models/mobileone_s2_unfused.pth.tar
```

So place the file there:

```bash
mkdir -p /content/uav-person-tracker/external/SiamAPN/SiamAPN++/pretrained_models
cp /path/to/mobileone_s2_unfused.pth.tar /content/uav-person-tracker/external/SiamAPN/SiamAPN++/pretrained_models/
```

### 5. Set The Competition Dataset Root

Edit:

`external/SiamAPN/SiamAPN++/experiments/config.yaml`

Set this field:

```yaml
DATASET:
    TYPE: 'competition'
    NAMES:
    - 'COMPETITION'

    COMPETITION:
        ROOT: '/content/path/to/your/competition_root'
        FRAME_RANGE: 100
        NUM_USE: -1
```

The `ROOT` directory should be the competition dataset root that contains the expected `metadata/contestant_manifest.json`.

### 6. Reduce The Config For A Smoke Test

For the first run, temporarily change these values in:

`external/SiamAPN/SiamAPN++/experiments/config.yaml`

Recommended smoke-test settings:

```yaml
TRAIN:
    EPOCH: 1
    START_EPOCH: 0
    BATCH_SIZE: 2
    NUM_GPU: 1
    NUM_WORKERS: 2
    PRINT_FREQ: 1

DATASET:
    COMPETITION:
        ROOT: '/content/path/to/your/competition_root'
        FRAME_RANGE: 20
        NUM_USE: 32
```

Why:

- `EPOCH: 1` keeps the run short
- `BATCH_SIZE: 2` is safer for first GPU validation
- `NUM_USE: 32` keeps the synthetic epoch small enough to reach checkpoint save quickly

### 7. Run The Training Smoke Test

From the upstream project root:

```bash
cd /content/uav-person-tracker/external/SiamAPN/SiamAPN++
uv run python tools/train_apn++.py --cfg experiments/config.yaml
```

### 8. What Success Looks Like

The smoke test is successful if you confirm all of these:

- model creation succeeds
- MobileOne-S2 pretrained weights load without crashing
- dataset creation succeeds
- training starts and prints losses
- backward pass runs
- a checkpoint is written under `snapshot/`

Expected output directories relative to:

`/content/uav-person-tracker/external/SiamAPN/SiamAPN++`

- logs: `./logs`
- checkpoints: `./snapshot`

### 9. Fast Debug Checks

If training fails, run these checks in Colab.

Check dataset root:

```bash
python - <<'PY'
import os, json
root = "/content/path/to/your/competition_root"
manifest = os.path.join(root, "metadata", "contestant_manifest.json")
print("root exists:", os.path.isdir(root))
print("manifest exists:", os.path.isfile(manifest))
if os.path.isfile(manifest):
    with open(manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("manifest keys:", list(data.keys())[:10])
PY
```

Check pretrained weights path:

```bash
python - <<'PY'
import os
path = "/content/uav-person-tracker/external/SiamAPN/SiamAPN++/pretrained_models/mobileone_s2_unfused.pth.tar"
print("exists:", os.path.isfile(path))
print("size bytes:", os.path.getsize(path) if os.path.isfile(path) else -1)
PY
```

Check one dataset sample:

```bash
cd /content/uav-person-tracker/external/SiamAPN/SiamAPN++
python - <<'PY'
from pysot.core.config_adapn import cfg
from pysot.datasets.dataset_competition_adapn import CompetitionTrkDataset

cfg.merge_from_file("experiments/config.yaml")
dataset = CompetitionTrkDataset()
sample = dataset[0]
for key, value in sample.items():
    shape = tuple(value.shape) if hasattr(value, "shape") else None
    print(key, type(value).__name__, shape)
PY
```

### 10. After The Smoke Test

If the smoke test passes, the next step is:

1. keep the faithful upstream path as the truth source
2. migrate only the used code into this repo
3. delete unused `external/` code after the migration is stable
