# SiamAPN++ Port Provenance

## Upstream Sources

- SiamAPN upstream repository: `https://github.com/vision4robotics/SiamAPN`
- Upstream implementation path used for this port: `SiamAPN++/`
- Local reference path before deletion: `external/SiamAPN/SiamAPN++/`
- Exact upstream snapshot: unknown from the vendored copy. Use the repository commit that added the vendored reference copy as the local source snapshot if exact upstream commit recovery is needed.
- MobileOne upstream repository: `https://github.com/apple/ml-mobileone`
- Local MobileOne reference path before deletion: `external/ml-mobileone/`

## Source To Active Mapping

- `external/SiamAPN/SiamAPN++/pysot/models/utile_adapn.py` -> `models/adapn.py`
- `external/SiamAPN/SiamAPN++/pysot/models/loss_adapn.py` -> `models/losses.py`
- `external/SiamAPN/SiamAPN++/pysot/datasets/anchortarget_adapn.py` -> `data/adapn_targets.py`
- `external/SiamAPN/SiamAPN++/pysot/tracker/adsiamapn_tracker.py` -> `app/tracking.py`
- `external/SiamAPN/SiamAPN++/pysot/models/model_builder_adapn.py` -> `models/siamapn.py`
- `external/SiamAPN/SiamAPN++/pysot/models/backbone/mobileone.py` -> `models/backbone/mobileone.py`
- `external/ml-mobileone/mobileone.py` -> `models/backbone/_mobileone.py`

## Adaptations

- Removed hardcoded `.cuda()` calls so tensors follow the active runtime device.
- Replaced global `pysot`/`yacs` `cfg` access with this repo's dataclass/YAML configuration.
- Integrated the active UAV competition dataset, bad-video skipping, and current `uv run train`, `uv run eval-train`, and `uv run eval-public` CLI flows.
- Kept runtime imports inside active repo modules; runtime code does not import from `external/`.
- Adapted MobileOne-S2 into this repo's package layout and preserved the original SiamAPN++ feature contract: template `[B, 384, 8, 8]`, `[B, 256, 6, 6]`; search `[B, 384, 28, 28]`, `[B, 256, 26, 26]`.
- Preserved original-style SiamAPN++ APN/head/loss/target/decode contracts: `APN`, `ClsAndLoc`, `cls1`, `cls2`, `cls3`, `loc`, `xff`/`ranchors`, `AnchorTarget`, `AnchorTarget3`, score fusion, penalties, Hanning window, best-candidate decode, and learning-rate state update.

## License And Notice Preservation

- SiamAPN++ reference code was distributed with Apache License 2.0 under `external/SiamAPN/SiamAPN++/LICENSE` before deletion.
- Apple ML-MobileOne reference code was distributed with the Apple license text under `external/ml-mobileone/LICENSE` before deletion.
- The license/notice text is preserved in `docs/references/third_party_notices.md` after deleting `external/`.

## Deletion Decision

The active implementation now carries the code needed at runtime, the source mapping is recorded here, and golden tests cover the active contracts that were ported. `external/` can be deleted without breaking runtime imports.

Remaining quality caveat: longer from-scratch SiamAPN++/MobileOne training and stronger evaluation are still needed to prove checkpoint quality. That is independent of whether the local reference copy is required for runtime.
