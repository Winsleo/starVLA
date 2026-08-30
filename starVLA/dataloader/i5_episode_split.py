# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Deterministic train/val/test split over LeRobot episodes for I5 (D-064).

D-064 fixes the split at 80/10/10 **by episode index within each task**, so that all 40 LIBERO tasks
appear in all three splits and no clip straddles an episode boundary. Two properties matter enough
to be enforced here rather than assumed by the caller:

* **Deterministic and RNG-free.** The split is a pure function of the episode inventory, so the
  offline latent cache and any later re-derivation agree without carrying a seed around.
* **Interleaved, not contiguous.** Within a task, episode `i` lands in a split by `i % 10`. A
  contiguous cut ("last 20% of each task") would be equally deterministic but would couple the split
  to whatever recording order the episode index encodes; interleaving removes that coupling and
  guarantees per-task balance. This is still "by episode index" in the sense D-064 fixes.

The eval split is generous by construction: about 170 held-out episodes at roughly 150 nine-frame
windows each is on the order of 25k clips.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

TRAIN = "train"
VAL = "val"
TEST = "test"
SPLITS = (TRAIN, VAL, TEST)

#: D-064's 80/10/10, expressed as the per-decade pattern the interleave walks.
_DECADE_PATTERN: tuple[str, ...] = (TRAIN,) * 8 + (VAL, TEST)


def split_of(position: int, pattern: tuple[str, ...] = _DECADE_PATTERN) -> str:
    """Split for the `position`-th episode of a task (0-based, in ascending episode index)."""
    if position < 0:
        raise ValueError(f"position must be >= 0, got {position}")
    return pattern[position % len(pattern)]


def assign_splits(
    episodes_by_task: Mapping[object, Iterable[int]],
    *,
    pattern: tuple[str, ...] = _DECADE_PATTERN,
    require_all_splits: bool = True,
) -> dict[int, str]:
    """Map every episode index to a split.

    Args:
        episodes_by_task: task identifier -> that task's episode indices, in any order.
        pattern: per-decade split pattern; defaults to D-064's 80/10/10.
        require_all_splits: raise if a task cannot populate all three splits. This is S1's
            "every task non-empty in all three splits" gate, enforced where the split is built so a
            too-small task cannot slip into the cache unnoticed.

    Returns:
        episode index -> split name.

    Raises:
        ValueError: on a duplicate episode index across tasks, or when `require_all_splits` is set
            and some task is too small to reach every split.
    """
    assignment: dict[int, str] = {}
    for task, episodes in sorted(episodes_by_task.items(), key=lambda item: str(item[0])):
        ordered = sorted(episodes)
        seen: dict[str, int] = {name: 0 for name in SPLITS}
        for position, episode in enumerate(ordered):
            if episode in assignment:
                raise ValueError(f"episode {episode} appears under more than one task")
            name = split_of(position, pattern)
            assignment[episode] = name
            seen[name] += 1
        if require_all_splits:
            empty = [name for name in SPLITS if seen[name] == 0]
            if empty:
                raise ValueError(
                    f"task {task!r} has only {len(ordered)} episode(s) and cannot populate "
                    f"{', '.join(empty)}; at least {len(pattern)} are needed for {pattern}"
                )
    return assignment


def split_counts(assignment: Mapping[int, str]) -> dict[str, int]:
    """Episode count per split, for cache metadata and provenance."""
    counts = {name: 0 for name in SPLITS}
    for name in assignment.values():
        counts[name] += 1
    return counts
