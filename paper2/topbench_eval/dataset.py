"""
Benchmark iterator for the EditCLIP-style evaluation.

Expected directory layout (EditCLIP benchmark convention):
    <root>/
        01/<task_name>/{train,test}/<id>_{0,1}.{jpg,png}
        02/<task_name>/{train,test}/...
        03/<task_name>/...

A pair is `(<id>_0, <id>_1)` = (source, edited). For each TEST source we
sample K random TRAIN pairs as exemplars and run the model
    output = Method( query=source_test, exemplar=(src_train, ed_train) )
then compare `output` against `edited_test`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class Pair:
    src_path: str
    ed_path: str
    pair_id: str

    @property
    def name(self) -> str:
        return self.pair_id


@dataclass
class Task:
    category: str       # "01" / "02" / "03"
    name: str           # "boy2girl" / "charcoal" / ...
    train_pairs: List[Pair]
    test_pairs: List[Pair]

    @property
    def full_id(self) -> str:
        return f"{self.category}/{self.name}"


@dataclass
class EvalSample:
    """One row of work: apply this task's edit to this test source.

    Attributes
    ----------
    task : owning Task
    query : test-source pair (query.src_path = the input; query.ed_path =
            the ground-truth target output)
    exemplars : K train pairs sampled to demonstrate the edit
    sample_idx : 0-based index of this exemplar replicate (for K>1 averaging)
    """
    task: Task
    query: Pair
    exemplars: List[Pair]
    sample_idx: int = 0

    @property
    def sample_id(self) -> str:
        ex_ids = "_".join(p.pair_id for p in self.exemplars)
        return (f"{self.task.category}__{self.task.name}__"
                f"q{self.query.pair_id}__ex{ex_ids}__s{self.sample_idx}")


# ---------------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------------

def _scan_pairs(folder: str) -> List[Pair]:
    """Group <id>_0.* and <id>_1.* files into Pair objects.

    Robust to mixed extensions inside the same folder. Skips any partial
    pair (only _0 or only _1 present).
    """
    if not os.path.isdir(folder):
        return []
    by_id: dict = {}
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in _IMG_EXTS:
            continue
        if "_" not in stem:
            continue
        pid, _, suffix = stem.rpartition("_")
        if suffix not in {"0", "1"}:
            continue
        slot = by_id.setdefault(pid, {})
        slot[suffix] = path

    pairs = []
    for pid in sorted(by_id):
        slot = by_id[pid]
        if "0" in slot and "1" in slot:
            pairs.append(Pair(src_path=slot["0"], ed_path=slot["1"], pair_id=pid))
    return pairs


def discover_tasks(benchmark_root: str,
                    categories: Optional[List[str]] = None,
                    task_filter: Optional[List[str]] = None) -> List[Task]:
    """Walk <benchmark_root>/<cat>/<task>/{train,test} and return Task list."""
    root = Path(benchmark_root)
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark root not found: {root}")

    tasks: List[Task] = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        if categories is not None and cat not in categories:
            continue
        for task_dir in sorted(cat_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            tname = task_dir.name
            if task_filter is not None and tname not in task_filter:
                continue
            train_pairs = _scan_pairs(str(task_dir / "train"))
            test_pairs = _scan_pairs(str(task_dir / "test"))
            if not train_pairs or not test_pairs:
                # Skip degenerate task with no usable split.
                continue
            tasks.append(Task(category=cat, name=tname,
                               train_pairs=train_pairs,
                               test_pairs=test_pairs))
    return tasks


# ---------------------------------------------------------------------------
# Sample stream
# ---------------------------------------------------------------------------

def iter_samples(tasks: List[Task],
                  k_exemplars: int = 1,
                  num_replicates: int = 1,
                  seed: int = 0,
                  max_tests_per_task: Optional[int] = None
                  ) -> Iterator[EvalSample]:
    """Yield EvalSample objects, one per (test pair, replicate index).

    Each sample has `k_exemplars` train pairs sampled WITHOUT replacement
    from the task's train set; the same test pair appears `num_replicates`
    times with different exemplar draws so metrics can be averaged.

    If the task has fewer than `k_exemplars` train pairs available, the
    sample is filled with replacement.
    """
    rng = random.Random(seed)
    for task in tasks:
        test_pairs = task.test_pairs
        if max_tests_per_task is not None:
            test_pairs = test_pairs[: max_tests_per_task]
        for query in test_pairs:
            for rep in range(num_replicates):
                pool = task.train_pairs
                if len(pool) >= k_exemplars:
                    chosen = rng.sample(pool, k_exemplars)
                else:
                    chosen = [rng.choice(pool) for _ in range(k_exemplars)]
                yield EvalSample(task=task, query=query,
                                  exemplars=chosen, sample_idx=rep)


def estimate_total_samples(tasks: List[Task],
                            num_replicates: int = 1,
                            max_tests_per_task: Optional[int] = None) -> int:
    n = 0
    for task in tasks:
        n_test = (len(task.test_pairs) if max_tests_per_task is None
                   else min(len(task.test_pairs), max_tests_per_task))
        n += n_test * num_replicates
    return n
