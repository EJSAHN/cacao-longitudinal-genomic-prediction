from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PhenotypeConfig:
    pseudo_count: float = 0.5
    time_order: dict[str, int] = field(
        default_factory=lambda: {
            "2016-II": 0,
            "2017-I": 1,
            "2017-II": 2,
            "2018-I": 3,
        }
    )


@dataclass
class GenotypeConfig:
    maf_min: float = 0.05
    call_rate_min: float = 0.95
    n_pcs: int = 10
    n_blocks: int = 5


@dataclass
class PredictionConfig:
    models: list[str] = field(default_factory=lambda: ["pc_ridge", "gblup"])
    n_splits_random: int = 5
    n_repeats_random: int = 20
    n_splits_inner: int = 4
    alpha_grid: list[float] = field(default_factory=lambda: [0.1, 10.0, 1000.0])
    n_permutations: int = 100
    max_targets: int | None = None


@dataclass
class SelectionIndexConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "healthy_pod_logit_intercept": 1.0,
            "healthy_pod_logit_slope": 0.5,
            "healthy_pod_logit_residual_sd": -0.25,
            "fpr_audpc_log_intercept": -1.0,
            "fpr_audpc_log_slope": -0.5,
            "fpr_audpc_log_residual_sd": -0.25,
            "flower_wbd_audpc_log_intercept": -1.0,
            "flower_wbd_audpc_log_slope": -0.5,
            "flower_wbd_audpc_log_residual_sd": -0.25,
            "branch_wbd_audpc_log_intercept": -1.0,
            "branch_wbd_audpc_log_slope": -0.5,
            "branch_wbd_audpc_log_residual_sd": -0.25,
        }
    )


@dataclass
class RunConfig:
    seed: int = 20250308
    reference: str = "both"
    phenotypes: PhenotypeConfig = field(default_factory=PhenotypeConfig)
    genotypes: GenotypeConfig = field(default_factory=GenotypeConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    selection_index: SelectionIndexConfig = field(default_factory=SelectionIndexConfig)


@dataclass
class PathConfig:
    accession_table_patterns: list[str] = field(default_factory=lambda: ["File_S1.xlsx", "File_S1.xls"])
    phenotype_table_patterns: list[str] = field(default_factory=lambda: ["File_S2.xlsx", "File_S2.xls"])
    sequencing_table_patterns: list[str] = field(default_factory=lambda: ["File_S3.xlsx", "File_S3.xls"])
    selection_table_patterns: list[str] = field(default_factory=lambda: ["File_S4.xlsx", "File_S4.xls"])
    criollo_vcf_patterns: list[str] = field(default_factory=lambda: ["*criollo*.vcf"])
    matina_vcf_patterns: list[str] = field(default_factory=lambda: ["*matina*.vcf"])
    tpg_supp_xlsx_patterns: list[str] = field(default_factory=lambda: ["tpg270069-sup-0002-suppmat.xlsx"])
    tpg_supp_docx_patterns: list[str] = field(default_factory=lambda: ["tpg270069-sup-0001-suppmat.docx"])


@dataclass
class AppConfig:
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def default_config() -> AppConfig:
    return AppConfig()


def load_config(config_path: Path | None) -> AppConfig:
    config = default_config()
    if config_path is None:
        return config

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    raw = asdict(config)
    raw["run"] = _deep_update(raw["run"], loaded)

    return AppConfig(
        run=RunConfig(
            seed=raw["run"]["seed"],
            reference=raw["run"]["reference"],
            phenotypes=PhenotypeConfig(**raw["run"]["phenotypes"]),
            genotypes=GenotypeConfig(**raw["run"]["genotypes"]),
            prediction=PredictionConfig(**raw["run"]["prediction"]),
            selection_index=SelectionIndexConfig(**raw["run"]["selection_index"]),
        ),
        paths=PathConfig(),
    )
