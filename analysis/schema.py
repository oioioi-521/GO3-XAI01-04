"""Shared result schemas and validation rules.

The project stores one row per image and metric in ``per_image.csv`` and one
row per experiment unit and metric in ``units.csv``.  Validation is kept in a
small standalone module so runners, merge scripts and analysis code can use
the same rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PER_IMAGE_COLUMNS = [
    "image_id",
    "dataset",
    "model",
    "method",
    "metric",
    "value",
    "time_ms",
]
UNIT_COLUMNS = [
    "method",
    "model",
    "dataset",
    "metric",
    "mean",
    "std",
    "n",
    "config_hash",
]
PER_IMAGE_KEY = ["image_id", "dataset", "model", "method", "metric"]
UNIT_KEY = ["method", "model", "dataset", "metric"]


@dataclass(frozen=True)
class ValidationSummary:
    rows: int
    methods: tuple[str, ...]
    models: tuple[str, ...]
    datasets: tuple[str, ...]
    metrics: tuple[str, ...]


class ResultValidationError(ValueError):
    """Raised when a result table would make downstream analysis unreliable."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ResultValidationError(f"{table} is missing required columns: {missing}")


def _require_nonempty_strings(frame: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    for column in columns:
        values = frame[column]
        invalid = values.isna() | values.astype(str).str.strip().eq("")
        if invalid.any():
            examples = frame.index[invalid].tolist()[:5]
            raise ResultValidationError(
                f"{table}.{column} contains blank values at rows {examples}"
            )


def _require_numeric_finite(
    frame: pd.DataFrame,
    columns: Iterable[str],
    table: str,
    *,
    allow_blank: bool = False,
) -> None:
    for column in columns:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if allow_blank:
            blank = original.isna() | original.astype(str).str.strip().eq("")
            invalid = ~blank & (numeric.isna() | ~np.isfinite(numeric))
        else:
            invalid = numeric.isna() | ~np.isfinite(numeric)
        if invalid.any():
            examples = frame.index[invalid].tolist()[:5]
            raise ResultValidationError(
                f"{table}.{column} contains non-finite values at rows {examples}"
            )


def _require_unique(frame: pd.DataFrame, key: list[str], table: str) -> None:
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, key].head(5).to_dict("records")
        raise ResultValidationError(f"{table} has duplicate keys: {examples}")


def _summary(frame: pd.DataFrame) -> ValidationSummary:
    def values(column: str) -> tuple[str, ...]:
        return tuple(sorted(frame[column].astype(str).unique()))

    return ValidationSummary(
        rows=len(frame),
        methods=values("method"),
        models=values("model"),
        datasets=values("dataset"),
        metrics=values("metric"),
    )


def validate_per_image(frame: pd.DataFrame) -> ValidationSummary:
    """Validate and summarize a per-image long result table."""

    _require_columns(frame, PER_IMAGE_COLUMNS, "per_image")
    if frame.empty:
        raise ResultValidationError("per_image is empty")
    _require_nonempty_strings(
        frame, ["image_id", "dataset", "model", "method", "metric"], "per_image"
    )
    _require_numeric_finite(frame, ["value"], "per_image")
    _require_numeric_finite(frame, ["time_ms"], "per_image", allow_blank=True)
    numeric_time = pd.to_numeric(frame["time_ms"], errors="coerce")
    if (numeric_time.dropna() < 0).any():
        raise ResultValidationError("per_image.time_ms cannot be negative")
    _require_unique(frame, PER_IMAGE_KEY, "per_image")
    return _summary(frame)


def validate_units(frame: pd.DataFrame) -> ValidationSummary:
    """Validate and summarize an aggregated unit table."""

    _require_columns(frame, UNIT_COLUMNS, "units")
    if frame.empty:
        raise ResultValidationError("units is empty")
    _require_nonempty_strings(frame, ["dataset", "model", "method", "metric"], "units")
    _require_numeric_finite(frame, ["mean", "std", "n"], "units")
    n = pd.to_numeric(frame["n"], errors="coerce")
    std = pd.to_numeric(frame["std"], errors="coerce")
    if (n <= 0).any() or ((n % 1) != 0).any():
        raise ResultValidationError("units.n must contain positive integers")
    if (std < 0).any():
        raise ResultValidationError("units.std cannot be negative")
    _require_unique(frame, UNIT_KEY, "units")
    return _summary(frame)


def read_csv(path: str | Path, table: str) -> pd.DataFrame:
    """Read a CSV with a clear error when the path is missing."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{table} CSV does not exist: {resolved}")
    return pd.read_csv(resolved)
