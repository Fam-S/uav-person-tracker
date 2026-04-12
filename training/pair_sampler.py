from __future__ import annotations

from dataclasses import dataclass
import random

from training.dataset_loader import SequenceFrame, UAV123Sequence


@dataclass(frozen=True)
class SiameseTrainingPair:
    sequence_name: str
    difficulty_tags: dict[str, bool]
    template: SequenceFrame
    search: SequenceFrame
    frame_gap: int


class UAV123PairSampler:
    def __init__(
        self,
        sequences: list[UAV123Sequence],
        max_frame_gap: int = 100,
    ) -> None:
        if not sequences:
            raise ValueError("Pair sampler requires at least one sequence.")
        if max_frame_gap < 1:
            raise ValueError("max_frame_gap must be at least 1.")

        self.sequences = [sequence for sequence in sequences if sequence.valid_frame_count >= 2]
        if not self.sequences:
            raise ValueError("Pair sampler requires sequences with at least two valid frames.")

        self.max_frame_gap = max_frame_gap

    def sample_pair(self, rng: random.Random | None = None) -> SiameseTrainingPair:
        rng = rng or random.Random()
        sequence = rng.choice(self.sequences)
        return self.sample_pair_from_sequence(sequence, rng=rng)

    def sample_pair_from_sequence(
        self,
        sequence: UAV123Sequence,
        rng: random.Random | None = None,
    ) -> SiameseTrainingPair:
        if sequence.valid_frame_count < 2:
            raise ValueError(f"Sequence {sequence.name} does not have enough valid frames.")

        rng = rng or random.Random()
        candidate_pairs: list[tuple[int, int]] = []
        for template_index in range(sequence.valid_frame_count - 1):
            template_frame = sequence.valid_frames[template_index]
            for search_index in range(template_index + 1, sequence.valid_frame_count):
                search_frame = sequence.valid_frames[search_index]
                gap = search_frame.frame_index - template_frame.frame_index
                if gap > self.max_frame_gap:
                    break
                candidate_pairs.append((template_index, search_index))

        if not candidate_pairs:
            raise ValueError(
                f"Sequence {sequence.name} does not contain valid frame pairs within max_frame_gap={self.max_frame_gap}."
            )

        template_index, search_index = rng.choice(candidate_pairs)
        template = sequence.valid_frames[template_index]
        search = sequence.valid_frames[search_index]
        return SiameseTrainingPair(
            sequence_name=sequence.name,
            difficulty_tags=sequence.difficulty_tags,
            template=template,
            search=search,
            frame_gap=search.frame_index - template.frame_index,
        )
