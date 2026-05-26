from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BARCODE_PATTERN = re.compile(r"^[ACGT]+(?:-[ACGT]+)?$")


def normalize_accession(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def extract_accession_from_vcf_sample(sample_name: str) -> str:
    tokens = sample_name.split("_")
    if tokens and BARCODE_PATTERN.fullmatch(tokens[-1]):
        tokens = tokens[:-1]
    accession = "_".join(tokens)
    normalized = normalize_accession(accession)
    if normalized == "CHOCO":
        return "CHOCOLATE"
    return normalized


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def percentile_interval(values: Iterable[float], alpha: float = 0.05) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return (np.nan, np.nan)
    lower = np.nanpercentile(array, 100 * (alpha / 2))
    upper = np.nanpercentile(array, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def safe_sheet_name(name: str, existing: set[str]) -> str:
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", name)[:31]
    if cleaned not in existing:
        existing.add(cleaned)
        return cleaned

    for suffix in range(1, 1000):
        candidate = f"{cleaned[:28]}_{suffix}"[:31]
        if candidate not in existing:
            existing.add(candidate)
            return candidate

    raise ValueError("Could not generate a unique sheet name.")
