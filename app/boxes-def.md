## Big Picture

These are **three different levels of representation** of the same object.

- **Target** = what the person actually occupies in the frame
- **Target crop / template crop** = a square image crop centered on that target, used as the tracker’s reference
- **Search crop** = a larger square crop around where we expect the target to be next

So:

- target = *object box*
- template crop = *reference patch*
- search crop = *where to look next*

---

## Step-by-Step Explanation

### 1. Target
This is the bbox around the actual person in the original frame.

Example:
- person occupies a box like `(x, y, w, h)`
- this box may be portrait-shaped
- this is the thing the user is really selecting

What it means:
- where the object is
- how big it is
- what its aspect ratio is

---

### 2. Target crop / Template crop
This is a **square crop** derived from the target.

It is usually:
- centered on the target
- includes some context around it
- resized to the fixed template input size used by the model

In a Siamese tracker, this becomes the **reference appearance**:
> “this is what I am tracking”

So the template crop is **not just the tight bbox**.  
It is usually a crop region around that bbox.

---

### 3. Search crop
This is another square crop, but larger.

It is:
- centered near the last known target position
- large enough to contain possible motion
- resized to the fixed search input size used by the model

In a Siamese tracker, this is where the model asks:
> “where inside this larger region does the target appear now?”

So the search crop is the local area for the next frame’s search.

---

## Example

Suppose in frame 1:

- the user selects a cyclist
- the cyclist bbox is tall and narrow

Then:

- **Target** = the cyclist bbox
- **Template crop** = a square patch around that cyclist, maybe with some background
- **Search crop** = a bigger square region in the next frame around where the cyclist is expected to be

The model compares:

- template crop features
against
- search crop features

---

## Common Pitfalls

### Pitfall 1: thinking target = template crop
Not exactly.

- target is the object box
- template crop is a square region derived from that target

### Pitfall 2: thinking search crop = full frame
Usually no.

For efficiency, the search crop is normally a **local region**, not the whole frame.

### Pitfall 3: thinking `127x127` and `255x255` describe the drawn frame boxes
They usually describe the **resized model inputs**, not literal screen boxes.

---

## In your app now

The current intended semantics are:

- **Target** = editable person box
- **Template Crop** = square crop derived from target
- **Search Crop** = larger square crop derived from template/target

That is the correct conceptual layering.