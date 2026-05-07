#!/usr/bin/env python3
"""Compute the fraction of events with a true tau vertex in the first N labels.

This script scans ROOT ntuples and checks whether the truth-vertex label
(`truth_tauVertex` by default) contains `truth_value` (default: 1) within the
first `N` entries (default: 10) for each event.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import awkward as ak
import numpy as np
import uproot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate fraction of events where the truth-vertex label appears "
            "in the first N entries of a truth-label branch."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input ROOT files and/or directories containing ROOT files.",
    )
    parser.add_argument(
        "--tree",
        default="CollectionTree",
        help="Name of the TTree to read (default: CollectionTree).",
    )
    parser.add_argument(
        "--branch",
        default="truth_tauVertex",
        help="Truth-label branch to inspect (default: truth_tauVertex).",
    )
    parser.add_argument(
        "--truth-value",
        type=int,
        default=1,
        help="Label value that marks the true vertex (default: 1).",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        nargs="+",
        default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        help=(
            "One or more leading-entry windows to inspect per event "
            "(default: 1-20). Example: --first-n 5 10 20"
        ),
    )
    parser.add_argument(
        "--step-size",
        default="250 MB",
        help="uproot chunk size, e.g. 100000, '250 MB', '1 GB' (default: 250 MB).",
    )
    parser.add_argument(
        "--denominator",
        choices=("all", "nonempty"),
        default="all",
        help=(
            "Denominator for fraction: 'all' uses all events, 'nonempty' uses only "
            "events with at least one truth-label entry."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directories for ROOT files.",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Also print per-file fractions.",
    )
    return parser.parse_args()


def collect_root_files(inputs: Iterable[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix == ".root":
            files.append(p)
            continue
        if p.is_dir():
            pattern = "**/*.root" if recursive else "*.root"
            files.extend(sorted(x for x in p.glob(pattern) if x.is_file()))
            continue
        raise FileNotFoundError(f"Input path does not exist or is unsupported: {p}")

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


def _as_list_per_event(arr: ak.Array) -> ak.Array:
    """Ensure an array has a per-event list axis so axis=1 operations are valid."""
    try:
        ak.num(arr, axis=1)
        return arr
    except Exception:
        # If branch is scalar per event, wrap each scalar as a length-1 list.
        return ak.singletons(arr)


def process_file(
    file_path: Path,
    tree: str,
    branch: str,
    first_n_list: list[int],
    truth_value: int,
    step_size: str,
    denominator: str,
) -> tuple[dict[int, int], int]:
    total_denominator = 0
    total_numerators = {n: 0 for n in first_n_list}

    spec = f"{file_path}:{tree}"
    for chunk in uproot.iterate(spec, [branch], step_size=step_size, library="ak"):
        labels = _as_list_per_event(chunk[branch])

        has_truth_vertex_map: dict[int, np.ndarray] = {}
        for n in first_n_list:
            in_window = labels[:, :n]
            has_truth_vertex_map[n] = ak.to_numpy(ak.any(in_window == truth_value, axis=1))

        if denominator == "nonempty":
            denom_mask = ak.to_numpy(ak.num(labels, axis=1) > 0)
            total_denominator += int(np.count_nonzero(denom_mask))
            for n in first_n_list:
                total_numerators[n] += int(
                    np.count_nonzero(has_truth_vertex_map[n] & denom_mask)
                )
        else:
            # All N values share the same event denominator.
            sample_n = first_n_list[0]
            total_denominator += int(len(has_truth_vertex_map[sample_n]))
            for n in first_n_list:
                total_numerators[n] += int(np.count_nonzero(has_truth_vertex_map[n]))

    return total_numerators, total_denominator

def plotting_script(x_values: list[int], y_values: list[float]) -> str:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title('Fraction of Events with True Tau Vertex in first-N Vertices')
    plt.xlabel('first-N vertices')
    plt.ylabel('Fraction of events with true tau vertex')
    plt.xticks(x_values)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("truth_vertex_fraction.png")
    plt.close()

def main() -> int:

    args = parse_args()

    first_n_values = sorted(set(args.first_n))
    if any(n <= 0 for n in first_n_values):
        raise ValueError("All --first-n values must be > 0")

    files = collect_root_files(args.inputs, recursive=args.recursive)
    if not files:
        raise FileNotFoundError("No .root files found in the provided inputs.")

    grand_num = {n: 0 for n in first_n_values}
    grand_den = 0

    for fpath in files:
        num_map, den = process_file(
            file_path=fpath,
            tree=args.tree,
            branch=args.branch,
            first_n_list=first_n_values,
            truth_value=args.truth_value,
            step_size=args.step_size,
            denominator=args.denominator,
        )
        for n in first_n_values:
            grand_num[n] += num_map[n]
        grand_den += den

        if args.per_file:
            frac_parts = []
            for n in first_n_values:
                frac = (num_map[n] / den) if den > 0 else float("nan")
                frac_parts.append(f"N={n}: {frac:.6f}")
            print(
                f"{fpath}: denominator={den}, "
                + ", ".join(frac_parts)
            )

    print("\nSummary")
    print(f"  files_scanned: {len(files)}")
    print(f"  tree: {args.tree}")
    print(f"  branch: {args.branch}")
    print(f"  truth_value: {args.truth_value}")
    print(f"  first_n_values: {first_n_values}")
    print(f"  denominator: {args.denominator}")
    print(f"  denominator_count: {grand_den}")

    numerators = [grand_num[n] for n in first_n_values]
    fractions = [
        (grand_num[n] / grand_den) if grand_den > 0 else float("nan")
        for n in first_n_values
    ]
    percents = [100.0 * f if np.isfinite(f) else float("nan") for f in fractions]

    print(f"  numerators: {numerators}")
    print(f"  fractions: {[round(f, 6) for f in fractions]}")
    print(f"  percents: {[round(p, 3) for p in percents]}")

    # Parse-friendly rows for scripting/plotting.
    print("\nPer-N results")
    for n, num, frac, pct in zip(first_n_values, numerators, fractions, percents):
        print(
            f"  N={n}: numerator={num}, denominator={grand_den}, "
            f"fraction={frac:.6f}, percent={pct:.3f}%"
        )

    plotting_script(first_n_values, fractions) # Call this plotting script (optional)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
