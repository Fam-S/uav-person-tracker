What I know now
From the actual data/raw package:
- root is data/raw/
- it contains dataset1 to dataset5 plus metadata/
- metadata/contestant_manifest.json is the best source of truth
- manifest paths match the files on disk
- total sequences on disk match the competition split:
  - train: 255
  - public_lb: 89
Important loading detail:
- train/annotation.txt contains one box per frame
- public_lb/annotation.txt contains only the init box for frame 0
- some boxes are decimals, so we should parse as float, not int
- 0,0,0,0 exists in train, so absence handling matters
What I would do next
Minimal first step: build one small clean loader around the manifest.
1. Create one simple competition-data module
I would add a single file, likely data/competition.py, with only these responsibilities:
- read contestant_manifest.json
- resolve file paths
- parse annotations
- expose train/public split records in a clean format
Minimal API:
@dataclass
class SequenceRecord:
    seq_id: str
    split: str
    dataset: str
    seq_name: str
    video_path: Path
    n_frames: int
    native_fps: int
    init_box_xywh: np.ndarray
    gt_boxes_xywh: np.ndarray | None
And one main function:
def load_sequences(raw_root: Path, split: str) -> list[SequenceRecord]:
    ...
Why this first:
- smallest useful layer
- easy to read
- everything else can build on it
- keeps competition-specific logic out of training code for now
2. Make the public split behavior explicit
For public_lb:
- load only the init box from annotation.txt
- set gt_boxes_xywh = None
- later use n_frames to generate submission rows
That avoids pretending public GT exists when it does not.
3. Add one tiny validation helper
Not a big preprocessing pipeline yet. Just a minimal verification function, something like:
def summarize_split(...)
It should print or return:
- number of sequences
- total frames
- a few sample sequence ids
- whether annotations line up with frame counts
This will help you trust the loader before training work starts.
4. Keep video decoding out of this first step
I would not build the full training dataset class yet.
First we should finish:
- manifest loader
- annotation parser
- split handling
- public leaderboard init-box handling
Then next step:
- frame sampling / video reading
- training pair dataset for the tracker
That is the clean “less is more” path.
After that
Once the loader is in place, the next minimal layer should be:
5. Build a small training dataset on top of SequenceRecord
Probably:
- one dataset class for train split only
- samples (template_frame, search_frame, boxes)
- uses cv2.VideoCapture directly at first
- no caching system unless it becomes necessary
I would avoid:
- big preprocessing
- cache manifests
- complicated registry abstractions
- multiple dataset classes too early
Recommended implementation order
1. data/competition.py
2. basic split summary / validation
3. public submission-id helper
4. train dataset class
5. only then model-facing sampling logic