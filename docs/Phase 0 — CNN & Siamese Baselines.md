# Phase 0 — CNN & Siamese Baselines (Start Here)

# What the Research Actually Says

> This page is grounded in **searched literature and real benchmark numbers**. Your SiamRPN++ + MobileOne-S0 idea is analyzed against actual published results below — including what the original SiamRPN++ authors tested, and where the real bottleneck lies.
> 

---

# ✅ Validating Your Idea: SiamRPN++ + MobileOne-S0

## What SiamRPN++ Paper Already Tested

The original CVPR 2019 SiamRPN++ paper **explicitly tested backbone swaps** — the authors replaced ResNet-50 with ResNet-18 and MobileNetV2, finding that both lighter variants achieved comparable accuracy while running at >70 FPS (vs 35 FPS for ResNet-50). This directly validates the backbone-swap approach you proposed.

On **UAV123**: SiamRPN++ (ResNet-50) achieves **AUC ~61.3%, Precision ~80.4%**. The MobileNetV2 variant falls ~1–3 AUC points below that based on VOT2018 ablations.

## What MobileOne-S0 Actually Looks Like

Real numbers from Apple's CVPR 2023 paper:

| Variant | Params | FLOPs | ImageNet Top-1 | iPhone12 Latency |
| --- | --- | --- | --- | --- |
| **MobileOne-S0** | **2.1 M** | **0.55 GFLOPs** | 71.4% | <1 ms |
| MobileOne-S1 | 4.8 M | 0.84 GFLOPs | 75.9% | <1 ms |
| MobileOne-S2 | 7.8 M | 1.26 GFLOPs | 77.4% | ~1.2 ms |
| MobileOne-S4 | 14.8 M | 2.98 GFLOPs | 79.4% | ~2.3 ms |

The re-parameterization trick (multi-branch training → single-path inference) means **zero extra latency at inference** compared to a plain conv — you get the training benefits free.

## ⚠️ The Real Risk: Is S0 Too Small?

MobileOne-S0's 71.4% ImageNet top-1 is close to MobileNetV2 (72.0%), but the architecture is shallower with fewer channels. For UAV tracking specifically, small targets at altitude require multi-scale semantic features that very shallow networks may not produce well.

**Recommendation based on the literature:** Start with S0 to validate the pipeline, but treat **MobileOne-S2 (~7.8M params, ~1.26 GFLOPs, 77.4% top-1)** as your actual submission backbone. It's still massively under budget and meaningfully more capable.

## Verdict on Your Idea

| Question | Answer |
| --- | --- |
| Is the SiamRPN++ backbone-swap approach valid? | ✅ Validated by original authors (MobileNetV2 tested in paper) |
| Does MobileOne work as a drop-in? | ✅ Same interface as MobileNetV2, just add reparameterize_model() at inference |
| Is S0 the best choice? | ⚠️ Probably not — S2 is safer for feature richness |
| Budget compliance? | ✅ Far under 30 GFLOPs and 50M params even at S4 |
| Expected UAV123 AUC (no fine-tuning)? | ~58–62% (S0/S2), needs fine-tuning to push higher |

---

# ⚡ The Real Bottleneck the Literature Points To: The Head

This is the most important finding from the search — and it's something your original idea doesn't address yet.

## Why Anchor-Based RPN Hurts for Small UAV Targets

Multiple papers (SiamAPN++, SiamCAR, SiamHSFT) document two concrete failure modes of anchor-based RPN on aerial data:

**1. Hyper-parameter sensitivity** — anchor scales/ratios must be hand-tuned per dataset. Default PySOT anchors target ground-level OTB/VOT objects. UAV targets are 10–50 px wide: you need much smaller anchor scales and more elongated ratios (vehicles from above look very different from pedestrians on the ground). Misconfigured anchors tank precision significantly.

**2. Class imbalance** — with tiny targets, most anchor proposals are background. RPN classification loss gets dominated by negatives unless you add OHEM or careful hard-negative mining.

## What the Anchor-Free Papers Actually Show on UAV123

From published results (all using AlexNet backbone for fair comparison):

| Tracker | Head Type | UAV123 AUC | UAV123 Prec | FPS |
| --- | --- | --- | --- | --- |
| SiamRPN++ | Anchor-based RPN | ~59% | ~79% | 100+ |
| SiamAPN | Adaptive anchor proposal | ~62% | ~83% | 40+ |
| **SiamAPN++** | **Attention + adaptive anchors** | **~64%** | **~86.7%** | 40+ |
| MobileTrack (custom MobileNet) | Anchor-free | 60.9% | 81.3% | 99.8 |

Same AlexNet backbone, different head → **SiamAPN++ beats SiamRPN++ by ~5 AUC points** on UAV123. The head matters as much as the backbone for this domain.

## What This Means for Your Idea

SiamRPN++ + MobileOne-S0 is a valid first experiment. But the literature suggests the higher-ceiling combination is:

> **SiamAPN++ + MobileOne-S2** — anchor-free adaptive head (UAV-tuned) + modern re-param backbone
> 

Replacing SiamAPN++'s default AlexNet with MobileOne-S2 is a natural contribution and the kind of "meaningful participant contribution" competition rules require.

---

# 📊 Full CNN Baseline Table (Real Published Numbers on UAV123)

| Tracker | Backbone | UAV123 AUC | UAV123 Prec | FPS | Head |
| --- | --- | --- | --- | --- | --- |
| SiamFC | AlexNet | 52.8% | 75.1% | 86 | Cross-correlation |
| DaSiamRPN | AlexNet | 58.6% | 79.6% | 110 | Anchor RPN |
| SiamRPN++ | AlexNet | ~59% | ~79% | 100+ | Anchor RPN |
| SiamRPN++ | MobileNetV2 | ~60% | ~78% | 70+ | Anchor RPN |
| **SiamRPN++ (ResNet-50)** | ResNet-50 | **61.3%** | **80.4%** | 35 | Anchor RPN |
| MobileTrack | Custom MobileNet | 60.9% | 81.3% | 99.8 | Anchor-free |
| SiamAPN | AlexNet | ~62% | ~83% | 40+ | Adaptive proposal |
| **SiamAPN++** | **AlexNet** | **~64%** | **~86.7%** | 40+ | **Attention + adaptive** |
| TCTrack++ | ResNet-50 | ~65% | 80.05% | 128 | Anchor-free + temporal |

**Realistic CNN ceiling with fine-tuning:** ~64–67% AUC. This is your target before switching to transformers.

---

# 🧪 The Two Experiments That Matter

Two variables are worth isolating: **backbone size** (is S0 enough?) and **head type** (does anchor-free matter?). Run these two in parallel — they answer both questions without redundancy.

## Experiment A — SiamRPN++ + MobileOne-S0 (your idea, as-is)

- Clone PySOT, swap in MobileOne-S0, run on val **zero fine-tuning**
- Anchor-based RPN + tiny backbone baseline
- ~2 hours to set up

## Experiment B — SiamAPN++ + MobileOne-S2 (research-suggested upgrade)

- Clone `vision4robotics/SiamAPN-plusplus`, swap AlexNet → MobileOne-S2
- Anchor-free head (proven +5 AUC on UAV123) + richer backbone
- Your actual competition CNN entry candidate

The gap between A and B tells you which lever matters more. If B beats A by >3 AUC, the head is the bottleneck. If they're close, the backbone is fine and fine-tuning will close the gap.

## Decision Gate

| Result | Action |
| --- | --- |
| Exp B AUC ≥ 63 | Submit CNN; run transformer in parallel for extra points |
| Exp B AUC 58–63 | Fine-tune B on competition data — likely gets you there |
| Exp B AUC < 58 | Switch focus to Aba-ViTrack (transformer track) |

---

# 🔺 Multi-Scale Features: FPN and Why It Matters Here

Yes — you're thinking of **FPN (Feature Pyramid Network)**, originally from object detection (Lin et al., CVPR 2017). The same idea applies directly to Siamese trackers, and it's especially relevant for UAV tracking where targets vary wildly in size.

## Why a Plain Backbone Struggles with Small

Loading a full tracker checkpoint across different architectures is an **architecture mismatch** — the heads are completely different. The only part that moves cleanly between any Siamese tracker is the **backbone**.

```
Full tracker checkpoint = backbone weights  +  head weights
                                  ↑                  ↑
                          transfers to any     architecture-specific,
                          tracker via           does NOT transfer
                          strict=False
```

So the right question is: which backbone weights should you start from?

## Safe Backbone Weights (no UAV data)

The competition dataset is almost certainly UAV123 + DTB70 + UAVTrack112 + UAV20L. Any checkpoint trained on those benchmarks would be contaminated. Load backbone weights only — initialized with ImageNet.

**Recommended:**

- `github.com/apple/ml-mobileone` → `mobileone_s2_unfused.pth.tar` (ImageNet only ✅)
- `torchvision.models.resnet50(weights='IMAGENET1K_V2')` (one line, 80.8% top-1 ✅)
- OpenMMLab SOT-modified ResNet-50 → strides already tuned for tracking ✅

**How to load backbone-only into SiamAPN++/SiamCAR (strict=False pattern):**

```python
state_dict = torch.load('mobileone_s2_unfused.pth.tar')['state_dict']
model.backbone.load_state_dict(state_dict, strict=False)
# strict=False loads matched layers, silently skips head layers
# Call reparameterize_model() before inference for MobileOne
```

**Avoid (contaminated with UAV benchmark data):**

| Source | Trained On | Safe? |
| --- | --- | --- |
| PySOT official `siamrpn_r50` | ILSVRC-VID + YouTube-BB + COCO + DET | ✅ Safe |
| Microsoft PySiamTracking models | GOT-10k + COCO + TrackingNet + LaSOT | ✅ Safe |
| SiamCAR `general_model.pth` | GOT-10k + LaSOT + OTB (no UAV data) | ✅ Safe |
| Any model explicitly mentioning UAV123 or DTB70 in training | UAV123, DTB70 | ❌ Contaminated |
| SiamAPN++ default released weights | Includes UAV benchmark data | ❌ Contaminated |

**What this means for you:** Use pretrained weights only as a **starting backbone initializer**, not as a final tracker. Load the weights, then fine-tune exclusively on the competition's 255 train sequences. This way you get the transfer learning benefit without contamination.

---

# 🔗 Key Repos

| Repo | Purpose |
| --- | --- |
| `github.com/STVIR/pysot` | SiamRPN++ official — supports backbone swap |
| `github.com/vision4robotics/SiamAPN-plusplus` | UAV-specific anchor-free Siamese |
| `github.com/apple/ml-mobileone` | MobileOne weights + `reparameterize_model()` |
| `github.com/vision4robotics/TCTrack` | Temporal CNN baseline, strong UAV numbers |
| `github.com/hqucv/siamban` | SiamBAN anchor-free, clean code |

---

> ⬆️ **After validating CNN baselines:** See the parent page for the Transformer strategy (Aba-ViTrack, HiT-Base). Use your best CNN score as the benchmark to beat.
>