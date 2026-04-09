## SiamLight Components

If we describe **SiamLight** as a lightweight **Siamese single-object tracker**, its main components are typically:

1. **Template branch**
   - Takes the target crop from the first frame
   - Encodes what the object looks like

2. **Search branch**
   - Takes the search region from the current frame
   - Looks for the target near the previous location

3. **Shared lightweight backbone**
   - Extracts features for both branches using the same network
   - In the comparison notes, this is described as a **MobileNet-V3-based** design

4. **Lightweight fusion / feature enhancement module**
   - Refines or fuses features before matching
   - This is one reason SiamLight is considered CPU-friendly

5. **Matching module**
   - Usually some form of **cross-correlation**
   - Compares template features with search features

6. **Prediction head**
   - Outputs a **classification score** for target location
   - Outputs **bounding box regression** for target size and position

7. **Tracking logic / post-processing**
   - Selects the best response
   - Applies smoothing, scale update, and crop update for the next frame

## Simple Flow

Template crop + search crop
-> shared lightweight feature extractor
-> feature fusion / correlation
-> score map + box prediction
-> updated target location

## In Short

A SiamLight-style tracker is basically:

- two inputs: template + search
- one shared lightweight backbone
- one lightweight matching stage
- one small prediction head

That is why it is attractive for **CPU-only edge tracking**.
