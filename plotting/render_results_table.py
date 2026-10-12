#!/usr/bin/env python3
"""Render a presentation-ready results table from CSV.

Example:
    python -m plotting.render_results_table \
        --source plotting/examples/preprocessing_augmentation_ablation.csv \
        --dest plotting/examples/preprocessing_augmentation_ablation.png \
        --title "Preprocessing and augmentation ablation — validation set, seed 0" \
        --maximize "3D Dice ↑" \
        --minimize "HD95 ↓"

The best value in each selected metric column is green and the worst is red.
Non-numeric cells are ignored when determining extrema.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HEADER_COLOR = "#244b63"
HEADER_TEXT_COLOR = "#ffffff"
BEST_COLOR = "#c9eac8"
WORST_COLOR = "#f4c7c3"
ROW_COLORS = ("#eaf2f8", "#fbf0e5", "#edf5e8", "#f3edf8", "#faebed")
GRID_COLOR = "#ffffff"
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def numeric_value(text: str) -> float | None:
    """Return the first finite number in a display cell, if present."""

    match = NUMBER_PATTERN.search(text.replace(",", ""))
    if match is None:
        return None
    value = float(match.group(0))
    return value if np.isfinite(value) else None


def load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"Expected a header and at least one data row in {path}")
    header = [cell.strip() for cell in rows[0]]
    body = [[cell.strip() for cell in row] for row in rows[1:] if any(cell.strip() for cell in row)]
    if any(len(row) != len(header) for row in body):
        raise ValueError("CSV rows do not all have the same number of columns")
    return header, body


def extrema_cells(
    header: list[str],
    rows: list[list[str]],
    maximize: list[str],
    minimize: list[str],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    overlap = set(maximize) & set(minimize)
    if overlap:
        raise ValueError(f"Columns cannot be both maximized and minimized: {sorted(overlap)}")
    missing = (set(maximize) | set(minimize)) - set(header)
    if missing:
        raise ValueError(f"Metric columns not found in CSV: {sorted(missing)}")

    best: set[tuple[int, int]] = set()
    worst: set[tuple[int, int]] = set()
    for column_name in [*maximize, *minimize]:
        column = header.index(column_name)
        parsed = [(row_index, numeric_value(row[column])) for row_index, row in enumerate(rows)]
        values = [(row_index, value) for row_index, value in parsed if value is not None]
        if not values:
            continue
        numbers = [value for _, value in values]
        best_value = max(numbers) if column_name in maximize else min(numbers)
        worst_value = min(numbers) if column_name in maximize else max(numbers)
        best.update((row_index, column) for row_index, value in values if np.isclose(value, best_value))
        # With only one unique result, marking it as both best and worst would
        # be misleading. Green takes precedence and no red cell is drawn.
        if not np.isclose(best_value, worst_value):
            worst.update((row_index, column) for row_index, value in values if np.isclose(value, worst_value))
    return best, worst


def column_widths(header: list[str]) -> list[float]:
    raw = []
    for index, name in enumerate(header):
        longest = max(len(name), 10)
        raw.append(max(1.25 if index else 2.4, min(3.2, longest * 0.12)))
    total = sum(raw)
    return [value / total for value in raw]


def render(
    header: list[str],
    rows: list[list[str]],
    title: str,
    destination: Path,
    maximize: list[str],
    minimize: list[str],
    dpi: int,
) -> None:
    best, worst = extrema_cells(header, rows, maximize, minimize)
    figure_height = 1.7 + 0.62 * (len(rows) + 1)
    fig, ax = plt.subplots(figsize=(14, figure_height), facecolor="white")
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)

    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellLoc="center",
        colLoc="center",
        colWidths=column_widths(header),
        bbox=[0.015, 0.02, 0.97, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (table_row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(1.4)
        if table_row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color=HEADER_TEXT_COLOR, weight="bold")
            continue

        data_position = (table_row - 1, column)
        cell.set_facecolor(ROW_COLORS[(table_row - 1) % len(ROW_COLORS)])
        if data_position in best:
            cell.set_facecolor(BEST_COLOR)
            cell.set_text_props(weight="bold")
        elif data_position in worst:
            cell.set_facecolor(WORST_COLOR)
            cell.set_text_props(weight="bold")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Input CSV")
    parser.add_argument("--dest", type=Path, required=True, help="Output PNG")
    parser.add_argument("--title", required=True)
    parser.add_argument("--maximize", nargs="*", default=[], metavar="COLUMN")
    parser.add_argument("--minimize", nargs="*", default=[], metavar="COLUMN")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    header, rows = load_csv(args.source)
    render(header, rows, args.title, args.dest, args.maximize, args.minimize, args.dpi)


if __name__ == "__main__":
    main()
