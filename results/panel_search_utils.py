"""Cell sampling and candidate-panel generation for ProtCell panel search."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import itertools
import math
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SearchSplit:
    search_indices: np.ndarray
    heldout_indices: np.ndarray
    stratum_counts: tuple[tuple[str, str, int], ...]


def stratified_search_split(
    obs: pd.DataFrame,
    donor_col: str,
    visit_col: str,
    cells_per_stratum: int,
    seed: int,
) -> SearchSplit:
    """Sample an equal number of search cells from every donor-visit stratum."""
    if cells_per_stratum <= 0:
        raise ValueError("cells_per_stratum must be positive")
    missing = [column for column in (donor_col, visit_col) if column not in obs]
    if missing:
        raise KeyError(f"Missing stratification columns: {missing}")
    frame = obs.loc[:, [donor_col, visit_col]].reset_index(drop=True)
    if frame.isna().any().any():
        raise ValueError("Stratification columns contain missing values")
    frame = frame.astype(str)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    counts: list[tuple[str, str, int]] = []
    grouped = frame.groupby([donor_col, visit_col], sort=True, observed=True)
    for (donor, visit), positions in grouped.indices.items():
        position_array = np.asarray(positions, dtype=np.int64)
        if len(position_array) < cells_per_stratum:
            raise ValueError(
                f"Stratum {(donor, visit)} must contain at least {cells_per_stratum} cells"
            )
        selected.append(
            np.asarray(
                rng.choice(position_array, size=cells_per_stratum, replace=False),
                dtype=np.int64,
            )
        )
        counts.append((str(donor), str(visit), int(len(position_array))))
    if not selected:
        raise ValueError("No donor-visit strata were found")
    search_indices = np.sort(np.concatenate(selected)).astype(np.int64, copy=False)
    heldout_indices = np.setdiff1d(
        np.arange(len(frame), dtype=np.int64), search_indices, assume_unique=True
    )
    if len(search_indices) + len(heldout_indices) != len(frame):
        raise RuntimeError("Search and heldout indices do not cover the dataset")
    if np.intersect1d(search_indices, heldout_indices).size:
        raise RuntimeError("Search and heldout indices overlap")
    return SearchSplit(search_indices, heldout_indices, tuple(counts))


def _ordered_panel(panel: Sequence[str], proteins: Sequence[str]) -> list[str]:
    order = {protein: index for index, protein in enumerate(proteins)}
    return sorted(set(panel), key=order.__getitem__)


def generate_random_panels(
    proteins: Sequence[str],
    budget: int,
    mandatory: Sequence[str],
    n_panels: int,
    seed: int,
) -> list[list[str]]:
    proteins = list(map(str, proteins))
    mandatory = _ordered_panel([protein for protein in mandatory if protein in proteins], proteins)
    if budget < len(mandatory) or budget > len(proteins):
        raise ValueError("budget must include mandatory markers and not exceed the protein count")
    optional = [protein for protein in proteins if protein not in mandatory]
    choose = budget - len(mandatory)
    total = math.comb(len(optional), choose)
    if n_panels > total:
        raise ValueError(f"Requested {n_panels} unique panels but only {total} exist")
    rng = np.random.default_rng(seed)
    if total <= 10000:
        combinations = list(itertools.combinations(optional, choose))
        selected = rng.choice(len(combinations), size=n_panels, replace=False)
        return [_ordered_panel(mandatory + list(combinations[index]), proteins) for index in selected]

    panels: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    while len(panels) < n_panels:
        sampled = rng.choice(optional, size=choose, replace=False).tolist()
        panel = _ordered_panel(mandatory + sampled, proteins)
        key = tuple(panel)
        if key not in seen:
            seen.add(key)
            panels.append(panel)
    return panels

