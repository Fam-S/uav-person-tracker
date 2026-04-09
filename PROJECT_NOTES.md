# Project Notes

This document keeps the longer project reasoning that was moved out of `README.md` to keep the main entry page shorter.

## Dataset and Training Strategy

For the college project, using a supplementary dataset is a good idea.

This project also overlaps with the **MTC-AIC4 competition**, so we tried to take advantage of that overlap where it is useful.

In practice, the competition dataset is used as a strong source for efficient aerial single-object tracking, while the college project extends beyond the competition scope toward a **GUI-based UAV person-tracking application**.

Why:

- the competition-style dataset is strong for **generic aerial single-object tracking**, but it is **not person-only**
- the available sequences span many categories, not just people
- the real target here is **UAV person tracking**, so a more human-focused dataset is a better fit for the application

So the recommended strategy is:

- do **not** replace the current aerial tracking dataset completely
- use it first for **generic UAV tracking pretraining**
- then add a more person-focused UAV dataset for specialization

### Best supplementary dataset

The best first supplementary dataset is **VisDrone**.

Why it stands out:

- real UAV imagery
- strong pedestrian / person presence
- difficult drone conditions such as small targets, clutter, and occlusion

### Recommended training stages

- **Stage 1:** train on the current aerial single-object tracking dataset for generic tracking behavior
- **Stage 2:** fine-tune on **VisDrone** for stronger UAV person-tracking performance

### Intended use of each stage

- The model after **Stage 1** can be used to make a **competition submission**
- The model after **Stage 2** is the one intended for the **college application**, since the project also requires a **GUI** and a practical **use case**

### Long-term tracking support

If long occlusion and recovery become more important, include **UAV20L-style** long sequences for evaluation or supplementary training.

## Final Product Vision

The final product is intended to be a **desktop GUI application** for UAV-assisted person tracking, not just a model demo.

The application should:

- load a live drone feed or recorded video
- let the user select **one person** in the first frame
- track that person in real time
- display a clear tracking state: **Tracking**, **Uncertain**, or **Lost**
- overlay the predicted bounding box, confidence score, and short trajectory trail
- save tracked video, frame-by-frame predicted boxes, and target lost / recovered events

### Suggested interface layout

- a main video panel
- a side control panel for loading video, selecting the target, and starting or pausing tracking
- a status panel for FPS, confidence, latency, and current tracker state
- an event log or timeline for target loss and recovery

### Possible use cases

Two practical use cases were considered:

- **UAV-assisted search and rescue support**: a software-level support tool where an operator selects a person from UAV footage and the system helps maintain visual tracking, report uncertainty, and preserve the last known position.
- **Outdoor Sports Filming Assistant**: the main selected use case for this project.

### Main selected use case: Outdoor Sports Filming Assistant

This use case focuses on helping a UAV operator or content creator keep one athlete centered during outdoor filming.

#### Scenario

Imagine a cyclist training on an outdoor trail.

Before takeoff or at the start of the recording, the operator opens the application and selects the athlete in the first frame. Once tracking starts, the system keeps following that person across the video while the UAV changes position and altitude.

During the session:

- the tracker keeps the athlete highlighted in the live view
- the GUI shows whether the system is **Tracking**, **Uncertain**, or **Lost**
- if the athlete passes behind trees, other riders, or scene clutter, the system shows reduced confidence instead of silently drifting
- if the athlete reappears clearly, tracking can recover and continue

The recorded result is not just a raw video stream. It becomes a guided filming tool that helps keep the subject visible and consistently framed.

#### Narrative use case

A small sports media team is filming a cyclist for a training highlight video.

Normally, the drone operator must manually keep the athlete centered while also managing drone movement, camera framing, and environmental obstacles. This is difficult when the subject moves quickly or becomes temporarily occluded by trees or terrain.

With this system, the operator selects the cyclist once, and the application continuously tracks the athlete in the live feed. The bounding box, confidence, and tracking state give immediate feedback about whether the subject is still being followed reliably. If the target becomes hard to see, the system warns the operator early, helping them correct the shot before losing the subject completely.

#### Benefits

- reduces the amount of manual camera correction needed during filming
- helps keep the selected athlete consistently framed in the shot
- gives the operator real-time feedback instead of relying only on visual judgment
- makes target loss more visible through explicit tracking states
- creates a clear and practical end-to-end application for the college project: model + GUI + real user scenario
