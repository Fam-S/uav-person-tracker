# App

## Purpose

This directory contains the desktop application layer for the project.

Version 1 is a **single-window desktop GUI for recorded-video person tracking**.

## Current V1 Status

The app has been implemented as a small `tkinter`-based package inside `app/`.

Current files:

- `main.py` - app entry point
- `config.py` - config loading and validation
- `controller.py` - app workflow, playback loop, and state handling
- `ui.py` - `tkinter` window, video canvas, controls, and status display
- `tracking.py` - backend interface, backend factory, and local backends
- `traditional_tracker.py` - stronger traditional tracker in a self-contained file
- `app_config.yaml` - app-level settings

## V1 Direction

The current product direction is:

- simple
- fast to build
- demo-focused
- recorded video only for version 1
- future support for both file input and live input

## Framework Choice

The selected framework for version 1 is **tkinter**.

Why:

- built into Python
- low setup overhead
- fast to build with
- good enough for a first demo application

## Current Layout

The current app layout has four parts:

1. **Main video panel**
   - shows the recorded video
   - displays the active tracking result directly on the frame

2. **Right-side control panel**
   - `Open Video`
   - `Select Target`
   - `Start`
   - `Pause`
   - `Reset`

3. **Bottom status strip**
   - FPS
   - confidence
   - latency
   - tracker state
   - active backend
   - latest event message
   - contextual hint line for the next action

4. **Event list**
   - recent app and tracking events

Recommended layout split:

- video area: about 70-75% of the window width
- right sidebar: about 25-30% of the window width
- bottom strip: thin full-width row below the main content

## Current Flow

The implemented V1 operator flow is:

`Open Video -> click target -> resize if needed -> Start -> Pause or Reset`

More specifically:

1. open a recorded video file
2. the app immediately enters selection mode on the first frame
3. click the target center to place a default target box
4. drag to resize the target box while keeping a fixed aspect ratio
5. start tracking
6. view overlays and status updates
7. pause or reset when needed

The `Select Target` button remains available as a reselect action.

## Current Overlay Output

The video view can show:

- tracked bounding box
- confidence text
- short trajectory trail
- a green `Target` box with an X during selection and after selection
- a square `Template Crop` overlay derived from the target box
- a square `Search Crop` overlay derived from the target box

## Tracker States

The app is explicit about tracking quality rather than silently drifting.

Current states include:

- `Tracking`
- `Uncertain`
- `Lost`
- `Paused`
- `Video Loaded`
- `Target Selected`

## Config File

Version 1 reads settings from:

- `app/app_config.yaml`

Main settings include:

- window size
- target FPS
- backend name
- optional checkpoint path
- uncertainty and lost thresholds
- trail length
- selection aspect ratio
- default selection size
- template crop scale
- search crop scale
- overlay toggles

The YAML intentionally keeps only the settings that are useful to change quickly.

Backend-specific tuning details stay in code defaults unless there is a strong reason to expose them later.

Example:

```yaml
tracking:
  backend: traditional_tracker
```

## Model Swapping Design

The GUI should remain easy to adapt to different tracking models.

To support that, the UI does not depend directly on one concrete tracker implementation. Instead, `controller.py` talks to a stable backend interface in `tracking.py`.

The interface is intentionally small:

```python
class TrackerBackend:
    def load(self, checkpoint: str | None) -> None: ...
    def initialize(self, frame, bbox) -> None: ...
    def track(self, frame): ...
    def reset(self) -> None: ...
```

This keeps the UI stable even if the underlying tracker changes later.

## Current Backend Options

The current V1 implementation includes three backend options selected from `app_config.yaml`:

- `traditional_tracker`
- `template_match`
- `mock`

### `traditional_tracker`

This is the current default backend.

It is a stronger traditional tracker implemented in `traditional_tracker.py` and stays fully contained inside `app/`.

Current improvements over the baseline tracker:

- better matching features using grayscale, gradient structure, and lighter HSV histogram similarity
- multi-scale search
- light motion prior using velocity
- stronger state logic using confidence, jump size, and score stability
- conservative candidate acceptance using initial-appearance and response-separation checks

The current implementation intentionally stays conservative:

- local search only
- frozen first-frame template
- no rotation template bank
- no re-detection stage yet
- no heavy smoothing layer over the displayed bbox

The internal tuning values for this backend currently live in `traditional_tracker.py` rather than in YAML, to keep `app_config.yaml` small and focused on high-value runtime settings.

### `template_match`

This is the simpler baseline backend.

It uses OpenCV template matching and keeps the app self-contained inside `app/`.

Why this is useful for V1:

- works immediately for the app demo flow
- does not depend on the rest of the repo
- preserves backend-swapping structure from the start

### `mock`

This backend moves the selected box in a small synthetic pattern.

Why it exists:

- helps test the GUI flow safely
- useful when debugging UI behavior separately from tracking behavior

## Overlay Semantics

The current selection and tracking overlays are intentionally separated into three concepts:

- `Target` - the user-edited person box
- `Template Crop` - a square crop derived from the target box
- `Search Crop` - a larger square crop derived from the template crop

This is closer to how Siamese-style trackers are usually reasoned about:

- the user selects a target extent in the frame
- the backend derives square crops from that target
- those crops are then resized internally for model input

Important note:

- the on-screen square overlays are **not literal pixel-sized 127x127 or 255x255 frame boxes**
- they are a visual representation of square template/search crop regions
- the exact crop logic is backend-dependent

## Running The App

Run from the repository root using the project environment:

```bash
uv run python -m app.main
```

The app expects the repo dependencies to be available, including:

- `opencv-python`
- `pillow`
- `pyyaml`

## Scope Limits For V1

Version 1 does **not** include:

- live camera input
- RTSP or drone stream input
- model selection in the GUI
- multi-object tracking
- advanced settings screens
- export or recording features

## Future Direction

The long-term app direction should support **both file input and live input**.

That can later expand to:

- recorded video files
- webcam or local camera input
- network or drone stream input

The first version does not need those paths yet.

## Main Use Case

The main selected use case remains:

- **Outdoor Sports Filming Assistant**

In that flow, an operator selects one athlete at the beginning, and the app helps keep that person visible and consistently framed during the session.
