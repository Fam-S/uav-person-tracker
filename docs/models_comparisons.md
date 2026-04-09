## Model Comparison

This document is focused on a **CPU-only single-object tracker** with these constraints:

- CPU only -> limited parallelism, latency-sensitive ops
- 30 FPS target -> about 33 ms per frame
- 720p input -> avoid full-frame processing
- Single object tracking -> template can be reused, but search still runs every frame
- Limited memory/cache -> large models hurt real latency

It also keeps a short reference section for broader edge models such as Jetson-oriented trackers and YOLO-based alternatives.

---

## 1. What Matters Most on CPU

For this problem, **real CPU latency** matters more than raw paper accuracy.

The best backbone is usually the one that gives:

- good spatial tracking features
- low real latency on CPU
- simple operators that export and optimize well
- low memory movement, not just low FLOPs

In practice, real-time CPU performance depends heavily on:

- Siamese tracking
- cropped search regions instead of full-frame processing
- a very small matching head
- deployment-friendly operators

That means **design choices can matter more than backbone swaps alone**.

---

## 2. Backbone Comparison for CPU-Only Tracking

## Direct Comparison: MobileOne-S1 vs ShuffleNetV2 1.0x vs MobileNetV3-Small

These three are all lightweight CNNs for mobile-edge vision, but they differ in latency, accuracy, and design philosophy.

Below is a practical ImageNet-style comparison focused on **accuracy vs latency vs FLOPs/params**.

| Model | Top-1 acc (ImageNet-1k) | Latency (typical mobile) | FLOPs / Params (sense only, not exact) | Key idea |
| :-- | :-- | :-- | :-- | :-- |
| **MobileOne-S1** | ~75.9-76% | **Sub-1 ms** on high-end phones such as iPhone 12 | Very low FLOPs via re-parameterization; about **4.8M params** | Re-parameterized conv blocks: multi-branch during training, merged for deployment speed |
| **ShuffleNetV2 1.0x** | ~69-73% | Low latency, but typically **>1 ms** on ARM and often slower than MobileOne-S1 at similar accuracy | Very low FLOPs and small parameter count | Pointwise group convs + channel shuffle with hardware-aware design rules |
| **MobileNetV3-Small** | ~67-70-72% depending on variant | Around **1-2 ms** on mobile chips, often strong on accelerator-friendly stacks | Very low FLOPs; about **3-5M params** depending on variant | NAS + h-swish + squeeze-and-excite for edge efficiency |

### Practical trade-offs

- **MobileOne-S1** usually has the best latency-accuracy trade-off of the three on modern phones and edge-style inference stacks.
- **ShuffleNetV2 1.0x** is extremely efficient and hardware-aware, but often gives up some accuracy versus MobileOne-S1.
- **MobileNetV3-Small** is the most balanced and ecosystem-friendly option, with strong tooling and widely available checkpoints.

### Design philosophy

- **MobileOne-S1**: use re-parameterization to make deployment faster than FLOPs alone would suggest.
- **ShuffleNetV2 1.0x**: optimize around real hardware rules such as balanced channel widths and low memory-access cost.
- **MobileNetV3-Small**: use NAS and advanced activation / attention choices to maximize accuracy for a small edge model.

### When to choose which

- Choose **MobileOne-S1** if ultra-low latency is the main goal and you want the strongest speed/accuracy trade-off.
- Choose **ShuffleNetV2 1.0x** if you want a simple, hardware-friendly design and care more about efficiency than squeezing out the last few accuracy points.
- Choose **MobileNetV3-Small** if you want the safest overall edge baseline with mature ecosystem support.

### MobileOne-S0 / S1

- Very promising for CPU-only deployment
- Inference is re-parameterized into mostly plain conv layers
- Good fit for ONNX / OpenVINO / other edge runtimes
- More attractive than many "efficient on paper" backbones because the inference graph is simple
- **MobileOne-S0** is the best first latency-oriented choice
- **MobileOne-S1** is the next step if S0 is too weak in difficult scenes

### ShuffleNetV2 (1.0x / 1.5x)

- One of the strongest practical CPU backbones
- Designed around real hardware efficiency, not just FLOPs
- Very good speed vs feature quality trade-off
- Strong fallback if MobileOne does not behave as expected in training or export

### MobileNetV3-Small / Large-0.75

- Safe, mature, and well-supported
- Still a good CPU-first option
- MobileNetV3-Small remains a strong lightweight baseline
- Usually easier to trust than heavier modern alternatives
- Weaker than stronger backbones in difficult scenes, but still very practical

### ConvNeXt-Tiny

- Strong representation
- Heavy (~28M params), high compute
- Typically too slow for this strict CPU target

### Light ViT (Tiny / MobileViT)

- Better global context
- Attention is usually inefficient on CPU
- Usually below 30 FPS without aggressive optimization
- Not a good first choice for this tracker

## Backbone Recommendation

**Primary backbone choice: MobileOne-S0**

Why:

- best combination of deployment-friendly ops and low-latency intent
- better fit for strict CPU inference than transformer-style choices
- stronger modern edge bet than relying only on MobileNetV3-Small

**Best fallback backbone: ShuffleNetV2-1.0x**

Why:

- very strong real-world efficiency
- excellent backup if MobileOne needs a simpler alternative

**Best mature baseline: MobileNetV3-Small**

Why:

- still one of the safest lightweight references
- useful as a known baseline even if it is not the new primary choice

---

## 3. Symmetric vs Asymmetric Siamese Branches

The two branches do **not** have to be identical.

That is valid because:

- the **template branch** runs once or rarely
- the **search branch** runs every frame

So a stronger template encoder can be almost free compared with making the search branch larger.

### Symmetric design

- Same backbone on template and search branches
- Easiest to train
- Easiest to export and debug
- Features align naturally for correlation-based matching
- Best first system to build and benchmark

### Asymmetric design

- Stronger template branch, lighter search branch
- Can improve target representation without fully paying the cost every frame
- More difficult to train because features may not align well
- Usually needs feature projection before matching

### If making the branches different

Recommended rules:

- keep the final embedding size the same on both branches
- add **1x1 projection layers** before matching
- normalize features before correlation
- keep the matching head simple

## Recommendation

Start with a **symmetric design**.

Only move to an asymmetric design after the symmetric model is benchmarked, because asymmetry adds training and matching risk even if it is conceptually appealing.

---

## 4. Architecture Candidates to Test

## Primary Architecture

**Symmetric MobileOne Siamese tracker**

- Template branch: **MobileOne-S0**
- Search branch: **MobileOne-S0**
- Template size: about **127x127**
- Search size: about **255x255** or **287x287**
- Matching: **depthwise cross-correlation**
- Head: small classification + box regression head
- Tracking style: crop around previous target location, do not process full 720p frame

Why this is the primary choice:

- strongest CPU-first backbone direction discussed so far
- symmetric branches reduce risk
- simple inference graph is good for CPU runtimes
- cleanest baseline for future comparisons

## Secondary Architecture

**Asymmetric MobileOne Siamese tracker**

- Template branch: **MobileOne-S1**
- Search branch: **MobileOne-S0**
- Add **1x1 projection** so both branches match in embedding size
- Normalize projected features before matching
- Keep the same lightweight correlation and prediction head

Why this is the secondary choice:

- template runs once, so it can be made stronger
- search branch stays fast
- good experiment if the symmetric S0 model lacks robustness in clutter or appearance change

Risk:

- feature alignment becomes harder than in the symmetric setup

---

## 5. Reference Tracker Families From Existing Work

These are still useful references, but they answer slightly different questions.

### Strict CPU-first tracker references

#### 1) SiamLight

- MobileNet-V3-based
- lightweight fusion modules
- explicitly aimed at reducing FLOPs and parameters
- still one of the most CPU-aligned literature directions

#### 2) SiamBAN with MobileNet / ShuffleNet

- anchor-free FCN tracker
- original paper reports 40 FPS
- a MobileNet/ShuffleNet version is a sensible lightweight direction
- actual speed depends a lot on the final backbone and head design

#### 3) LightTrack

- very efficient design
- paper reports **1.97M parameters**, **530M FLOPs**, and **38.4 FPS** on Snapdragon 845 Adreno 630 GPU
- useful lightweight reference, but its published speed is on mobile GPU, not strict CPU

### Best overall edge balance on accelerated edge hardware

#### 1) LiteTrack-B6 / B8

- strongest balanced option for Jetson-class hardware in these notes
- **LiteTrack-B6** reports **68.7 GOT-10k / 64.6 LaSOT / 80.8 TrackingNet**
- about **31.5 FPS on Orin NX** and **50.6 FPS with ONNX on Xavier NX**
- better evidence for accelerated edge than for CPU-only deployment

#### 2) LightTrack

- mature lightweight baseline
- strong edge-oriented reference even if it is older

#### 3) NanoTrack

- best when speed and size matter more than accuracy
- repo reports **MobileNetV3-based models around 2-3.4 MB**
- reports **>120 FPS on Apple M1 CPU**
- lower accuracy than LiteTrack, but very attractive for tiny deployments

---

## 6. YOLO as a Detector + Tracker Alternative

YOLO can work well for real-time video on CPU, but for tracking it is usually a **detector + tracker** pipeline rather than a pure tracker.

That makes it useful when you need:

- re-detection after occlusion
- recovery when the target leaves and re-enters the frame

For **small size + CPU speed**, the ranking from the earlier notes is still:

1. **YOLO26n**
2. **YOLO11n**
3. **YOLOv10n**
4. **YOLOv8n**

Key notes:

- **YOLO26n**: **2.4M params**, **5.4B FLOPs**, **38.9 ms CPU ONNX** at 640
- **YOLO11n**: **2.6M params**, **6.5B FLOPs**, **56.1 ms CPU ONNX** at 640
- **YOLOv8n**: **3.2M params**, **8.7B FLOPs**, **80.4 ms CPU ONNX** at 640
- **YOLOv10n**: strong efficiency-oriented design, but less directly pinned here with the same CPU metric style

For this tracker, YOLO is a better **secondary system direction** than a primary one, because per-frame detector cost is high for a 30 FPS CPU-only target.

---

## 7. Final Recommendations

### If the target is strict CPU-only single-object tracking

- **Primary backbone:** **MobileOne-S0**
- **Primary architecture:** **symmetric MobileOne-S0 Siamese tracker**
- **Secondary architecture:** **template MobileOne-S1 + search MobileOne-S0 asymmetric Siamese tracker**

### If MobileOne underperforms or is difficult to deploy

- switch to **ShuffleNetV2-1.0x** as the most practical backup

### If you want a literature-aligned CPU reference

- use **SiamLight** as the closest CPU-friendly reference direction

### If you need re-detection rather than pure local tracking

- use **YOLO26n + tracker** as the better alternative pipeline

### If you later target Jetson-class accelerated edge

- **LiteTrack-B6** becomes the strongest overall balance from these notes
