#!/usr/bin/env python3
"""Merge selected technology/preprocessing entries from metadata JSON files."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable


Metadata = dict[str, Any]


def parse_selection(selection: str) -> tuple[str, str]:
    """Parse an exact ``TECHNOLOGY=PREPROCESS`` selection."""
    if selection.count("=") != 1:
        raise ValueError(
            f"Invalid selection {selection!r}; expected TECHNOLOGY=PREPROCESS"
        )
    technology, preprocess = (part.strip() for part in selection.split("="))
    if not technology or not preprocess:
        raise ValueError(
            f"Invalid selection {selection!r}; expected TECHNOLOGY=PREPROCESS"
        )
    return technology, preprocess


def validate_shards(dataset_name: str, dataset: Metadata) -> None:
    """Require each shard's declared cell count to equal its index range."""
    for shard in dataset.get("shards", []):
        try:
            start = shard["start"]
            end = shard["end"]
            n_cells = shard["n_cells"]
        except KeyError as error:
            raise ValueError(
                f"Dataset {dataset_name!r} has a shard missing {error.args[0]!r}"
            ) from error
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (start, end, n_cells)):
            raise ValueError(f"Dataset {dataset_name!r} has non-integer shard bounds")
        if n_cells != end - start:
            raise ValueError(
                f"Dataset {dataset_name!r} has shard n_cells={n_cells}, "
                f"but end - start={end - start}"
            )


def assert_compatible_vocab(reference: Metadata, candidate: Metadata) -> None:
    """Ensure sources use one identical protein-to-index representation."""
    for key in ("global_vocab", "protein_to_id"):
        if key not in candidate:
            raise ValueError(f"Source metadata is missing {key!r}")
        if reference[key] != candidate[key]:
            raise ValueError(f"Source metadata has incompatible {key}")


def merge_metadata_sources(sources: Iterable[tuple[Metadata, str]]) -> Metadata:
    """Return metadata containing the datasets selected from ordered sources.

    Args:
        sources: Ordered ``(metadata, 'TECHNOLOGY=PREPROCESS')`` pairs.
    """
    sources = list(sources)
    if not sources:
        raise ValueError("At least one metadata source is required")

    reference, _ = sources[0]
    if "global_vocab" not in reference or "protein_to_id" not in reference:
        raise ValueError("First source metadata must contain global_vocab and protein_to_id")
    merged = copy.deepcopy(reference)
    merged["files"] = {}

    for metadata, selection in sources:
        assert_compatible_vocab(reference, metadata)
        technology, preprocess = parse_selection(selection)
        selected = [
            (name, dataset)
            for name, dataset in metadata.get("files", {}).items()
            if dataset.get("technology") == technology
            and dataset.get("preprocess") == preprocess
        ]
        if not selected:
            raise ValueError(
                f"Selection {selection!r} matched no datasets in a source metadata file"
            )
        for name, dataset in selected:
            if name in merged["files"]:
                raise ValueError(f"Duplicate dataset name while merging: {name!r}")
            validate_shards(name, dataset)
            merged["files"][name] = copy.deepcopy(dataset)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge selected technology/preprocessing entries from metadata JSON files."
    )
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--select", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.source) != len(args.select):
        parser.error("--source and --select must be supplied in matching pairs")

    sources = [
        (json.loads(path.read_text()), selection)
        for path, selection in zip(args.source, args.select)
    ]
    merged = merge_metadata_sources(sources)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()
