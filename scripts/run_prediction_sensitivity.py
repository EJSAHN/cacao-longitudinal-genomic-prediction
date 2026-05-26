#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Longitudinal genomic prediction analyses for repeated-harvest cacao disease traits.

Run from the project root, for example:

    python scripts/run_prediction_sensitivity.py --root . --output results_sensitivity --n-permutations 10000

The script reuses the local cacao_resilience_pipeline package and adds sensitivity
and reproducibility analyses used to support the main prediction workflow:

1) high-resolution blocked-CV permutation tests with BH-FDR q-values;
2) alternative relatedness-block definitions: Ward on markers, k-means on marker PCs,
   and k-means on genomic-relationship eigenvectors;
3) marker physical coverage and SNP spacing summaries;
4) post-QC missingness summaries and KNN-imputation sensitivity;
5) eigenvalue-based effective genomic dimensionality summaries;
6) software-version record for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold, KFold


# -----------------------------------------------------------------------------
# Project import handling
# -----------------------------------------------------------------------------


def add_project_package_to_path(root: Path) -> None:
    """Add local package folders to sys.path."""
    candidates = [
        root,
        root / "cacao_resilience_pipeline",
        root / "cacao_resilience_pipeline_robustness",
    ]
    for candidate in candidates:
        if (candidate / "cacao_resilience_pipeline" / "__init__.py").exists():
            sys.path.insert(0, str(candidate.resolve()))
            return
        if (candidate / "__init__.py").exists() and candidate.name == "cacao_resilience_pipeline":
            sys.path.insert(0, str(candidate.parent.resolve()))
            return


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str = "pearson") -> float:
    from scipy.stats import pearsonr, spearmanr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    a = y_true[mask]
    b = y_pred[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(a, b).correlation)
    return float(pearsonr(a, b).statistic)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"pearson_r": np.nan, "spearman_rho": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "pearson_r": safe_corr(y_true, y_pred, "pearson"),
        "spearman_rho": safe_corr(y_true, y_pred, "spearman"),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - np.nanmean(values)) / sd


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values, robust to NaNs."""
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    p_sub = p[finite]
    m = len(p_sub)
    order = np.argsort(p_sub)
    ranked = p_sub[order]
    raw = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    q[finite] = out
    return q


PREFERRED_TARGETS = [
    "legacy_healthy_pod_rate",
    "legacy_fpr_audpc_log",
    "legacy_flower_wbd_audpc_log",
    "legacy_branch_wbd_audpc_log",
    "healthy_pod_logit_intercept",
    "healthy_pod_logit_slope",
    "fpr_audpc_log_intercept",
    "fpr_audpc_log_slope",
    "flower_wbd_audpc_log_intercept",
    "flower_wbd_audpc_log_slope",
    "branch_wbd_audpc_log_intercept",
    "branch_wbd_audpc_log_slope",
]


def target_class(target: str) -> str:
    if target.startswith("legacy_"):
        return "legacy_static_or_total"
    if target.endswith("_intercept"):
        return "trajectory_intercept"
    if target.endswith("_slope"):
        return "trajectory_slope"
    return "other"


def disease_class(target: str) -> str:
    t = target.lower()
    if "branch_wbd" in t:
        return "Branch WBD"
    if "flower_wbd" in t:
        return "Flower WBD"
    if "fpr" in t:
        return "FPR"
    if "healthy" in t:
        return "Healthy pod"
    if "total_pods" in t:
        return "Total pods"
    return "Other"


# -----------------------------------------------------------------------------
# Kernel ridge / GBLUP CV and permutation
# -----------------------------------------------------------------------------


def select_alpha_kernel(K: np.ndarray, y: np.ndarray, alpha_grid: list[float], n_splits_inner: int, seed: int) -> float:
    n = len(y)
    n_splits = max(2, min(int(n_splits_inner), n))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_alpha = float(alpha_grid[0])
    best_score = -np.inf
    for alpha in alpha_grid:
        pred = np.full(n, np.nan)
        for train_idx, test_idx in splitter.split(K):
            K_train = K[np.ix_(train_idx, train_idx)]
            K_test = K[np.ix_(test_idx, train_idx)]
            model = KernelRidge(alpha=float(alpha), kernel="precomputed")
            model.fit(K_train, y[train_idx])
            pred[test_idx] = model.predict(K_test)
        score = safe_corr(y, pred, "pearson")
        if not np.isfinite(score):
            score = -np.inf
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def blocked_cv_gblup(
    y: np.ndarray,
    X_marker: np.ndarray,
    block_ids: np.ndarray,
    alpha_grid: list[float],
    n_splits_inner: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, list[float]]:
    """Outer blocked CV with fold-specific alpha selection, matching the original workflow style."""
    y = np.asarray(y, dtype=float)
    K = (X_marker @ X_marker.T) / X_marker.shape[1]
    groups = np.asarray(block_ids)
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=len(unique_groups))
    oof = np.full(len(y), np.nan)
    alphas = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X_marker, y, groups), start=1):
        K_train = K[np.ix_(train_idx, train_idx)]
        K_test = K[np.ix_(test_idx, train_idx)]
        alpha = select_alpha_kernel(K_train, y[train_idx], alpha_grid, n_splits_inner, seed + fold_id)
        model = KernelRidge(alpha=float(alpha), kernel="precomputed")
        model.fit(K_train, y[train_idx])
        oof[test_idx] = model.predict(K_test)
        alphas.append(alpha)
    metrics = regression_metrics(y, oof)
    metrics["alpha_median"] = float(np.nanmedian(alphas)) if alphas else np.nan
    metrics["alpha_mean"] = float(np.nanmean(alphas)) if alphas else np.nan
    return metrics, oof, alphas


def fixed_alpha_blocked_cv_stat(
    y: np.ndarray,
    K: np.ndarray,
    block_ids: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    groups = np.asarray(block_ids)
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(K, y, groups):
        K_train = K[np.ix_(train_idx, train_idx)]
        K_test = K[np.ix_(test_idx, train_idx)]
        model = KernelRidge(alpha=float(alpha), kernel="precomputed")
        model.fit(K_train, y[train_idx])
        oof[test_idx] = model.predict(K_test)
    return safe_corr(y, oof, "pearson"), oof


def permutation_test_fast(
    y: np.ndarray,
    X_marker: np.ndarray,
    block_ids: np.ndarray,
    alpha: float,
    n_permutations: int,
    seed: int,
) -> dict[str, float]:
    """Fast fixed-alpha blocked-CV permutation using precomputed fold linear maps."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    K = (X_marker @ X_marker.T) / X_marker.shape[1]
    groups = np.asarray(block_ids)
    splitter = GroupKFold(n_splits=len(np.unique(groups)))

    fold_maps = []
    for train_idx, test_idx in splitter.split(K, y, groups):
        K_train = K[np.ix_(train_idx, train_idx)]
        K_test = K[np.ix_(test_idx, train_idx)]
        # Prediction for a fixed alpha is linear in y_train:
        # pred_test = K_test_train @ inv(K_train + alpha I) @ y_train.
        A = K_train + float(alpha) * np.eye(len(train_idx))
        try:
            B = np.linalg.solve(A.T, K_test.T).T
        except np.linalg.LinAlgError:
            B = K_test @ np.linalg.pinv(A)
        fold_maps.append((train_idx, test_idx, B))

    def eval_y(y_vec: np.ndarray) -> float:
        oof = np.full(len(y_vec), np.nan)
        for train_idx, test_idx, B in fold_maps:
            oof[test_idx] = B @ y_vec[train_idx]
        return safe_corr(y_vec, oof, "pearson")

    observed = eval_y(y)
    null_values = np.empty(int(n_permutations), dtype=float)
    for i in range(int(n_permutations)):
        null_values[i] = eval_y(rng.permutation(y))
    # One-sided empirical P for positive predictive accuracy, matching previous logic.
    empirical_p = (1.0 + np.sum(null_values >= observed)) / (1.0 + len(null_values))
    return {
        "permutation_observed_pearson_r": float(observed),
        "null_mean": float(np.nanmean(null_values)),
        "null_sd": float(np.nanstd(null_values, ddof=0)),
        "null_q025": float(np.nanquantile(null_values, 0.025)),
        "null_q500": float(np.nanquantile(null_values, 0.500)),
        "null_q975": float(np.nanquantile(null_values, 0.975)),
        "empirical_p_value": float(empirical_p),
        "n_permutations": int(n_permutations),
    }


# -----------------------------------------------------------------------------
# Blocking strategies
# -----------------------------------------------------------------------------


def effective_n_blocks(requested: int, n: int) -> int:
    # Mirrors the original pipeline guard against tiny block sizes.
    k = max(2, min(int(requested), n // 5))
    if k >= n:
        k = max(2, n // 2)
    return int(k)


def make_blocks(
    X_marker: np.ndarray,
    pcs: np.ndarray,
    method: str,
    n_blocks: int,
    seed: int,
) -> np.ndarray:
    n = X_marker.shape[0]
    k = effective_n_blocks(n_blocks, n)
    if method == "ward_marker_euclidean":
        distances = pdist(X_marker, metric="euclidean")
        labels = fcluster(linkage(distances, method="ward"), k, criterion="maxclust")
        return labels.astype(int)
    if method == "kmeans_pc":
        p = pcs[:, : min(10, pcs.shape[1])]
        labels = KMeans(n_clusters=k, random_state=seed, n_init=50).fit_predict(p)
        return (labels + 1).astype(int)
    if method == "kmeans_grm_eigenvectors":
        K = (X_marker @ X_marker.T) / X_marker.shape[1]
        evals, evecs = np.linalg.eigh(K)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        p = evecs[:, : min(10, evecs.shape[1])]
        labels = KMeans(n_clusters=k, random_state=seed, n_init=50).fit_predict(p)
        return (labels + 1).astype(int)
    raise ValueError(f"Unknown blocking method: {method}")


def block_summary(reference: str, method: str, n_requested: int, accessions: list[str], block_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"accession_norm": accessions, "block_id": block_ids.astype(int)})
    counts = frame.groupby("block_id", as_index=False).size().rename(columns={"size": "n_accessions"})
    rows = []
    for _, row in counts.iterrows():
        rows.append({
            "reference": reference,
            "block_method": method,
            "requested_n_blocks": n_requested,
            "actual_n_blocks": int(len(counts)),
            "block_id": int(row["block_id"]),
            "n_accessions": int(row["n_accessions"]),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Marker QC summaries
# -----------------------------------------------------------------------------


def marker_map_coverage(reference: str, filtered_variant_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = filtered_variant_table.copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    rows = []
    spacing_rows = []
    for chrom, frame in df.dropna(subset=["pos"]).groupby("chrom", sort=False):
        pos = np.sort(frame["pos"].to_numpy(dtype=float))
        if len(pos) >= 2:
            spacing = np.diff(pos)
            mean_spacing = float(np.mean(spacing))
            median_spacing = float(np.median(spacing))
            p90_spacing = float(np.quantile(spacing, 0.90))
            max_gap = float(np.max(spacing))
        else:
            spacing = np.array([], dtype=float)
            mean_spacing = median_spacing = p90_spacing = max_gap = np.nan
        rows.append({
            "reference": reference,
            "chrom": chrom,
            "n_snps": int(len(pos)),
            "min_pos": int(np.min(pos)) if len(pos) else np.nan,
            "max_pos": int(np.max(pos)) if len(pos) else np.nan,
            "covered_span_bp": int(np.max(pos) - np.min(pos) + 1) if len(pos) else np.nan,
            "mean_spacing_bp": mean_spacing,
            "median_spacing_bp": median_spacing,
            "p90_spacing_bp": p90_spacing,
            "max_gap_bp": max_gap,
        })
        if len(spacing):
            spacing_rows.extend(
                {"reference": reference, "chrom": chrom, "spacing_bp": float(s)} for s in spacing
            )
    by_chrom = pd.DataFrame(rows)
    total = pd.DataFrame([
        {
            "reference": reference,
            "chrom": "__TOTAL__",
            "n_snps": int(by_chrom["n_snps"].sum()),
            "min_pos": np.nan,
            "max_pos": np.nan,
            "covered_span_bp": float(by_chrom["covered_span_bp"].sum()),
            "mean_spacing_bp": float(pd.DataFrame(spacing_rows)["spacing_bp"].mean()) if spacing_rows else np.nan,
            "median_spacing_bp": float(pd.DataFrame(spacing_rows)["spacing_bp"].median()) if spacing_rows else np.nan,
            "p90_spacing_bp": float(pd.DataFrame(spacing_rows)["spacing_bp"].quantile(0.90)) if spacing_rows else np.nan,
            "max_gap_bp": float(pd.DataFrame(spacing_rows)["spacing_bp"].max()) if spacing_rows else np.nan,
        }
    ])
    return pd.concat([by_chrom, total], ignore_index=True), pd.DataFrame(spacing_rows)


def effective_dimension_summary(reference: str, X_marker: np.ndarray) -> pd.DataFrame:
    K = (X_marker @ X_marker.T) / X_marker.shape[1]
    evals = np.linalg.eigvalsh(K)
    evals = np.clip(evals, 0, None)
    evals_desc = np.sort(evals)[::-1]
    total = float(np.sum(evals_desc))
    if total <= 0:
        return pd.DataFrame([{"reference": reference}])
    cum = np.cumsum(evals_desc) / total
    participation_ratio = float(total**2 / np.sum(evals_desc**2)) if np.sum(evals_desc**2) > 0 else np.nan
    rows = [{
        "reference": reference,
        "n_accessions": int(X_marker.shape[0]),
        "n_markers": int(X_marker.shape[1]),
        "rank_nonzero_eigenvalues": int(np.sum(evals_desc > 1e-10)),
        "effective_marker_dimension_participation_ratio": participation_ratio,
        "n_components_80pct": int(np.searchsorted(cum, 0.80) + 1),
        "n_components_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "n_components_95pct": int(np.searchsorted(cum, 0.95) + 1),
        "n_components_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "top1_eigen_fraction": float(evals_desc[0] / total),
        "top5_eigen_fraction": float(np.sum(evals_desc[:5]) / total),
        "top10_eigen_fraction": float(np.sum(evals_desc[:10]) / total),
    }]
    return pd.DataFrame(rows)


def filtered_sample_missingness(
    reference: str,
    vcf_path: Path,
    phenotype_data,
    genotype_config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from cacao_resilience_pipeline.genotypes import (
        build_sample_table,
        compute_variant_qc,
        filter_variants,
        read_vcf_dosage,
    )

    variant_table, sample_names, dosage_matrix = read_vcf_dosage(vcf_path)
    sample_table = build_sample_table(sample_names, phenotype_data.accession_table, phenotype_data.sequencing_summary)
    variant_qc = compute_variant_qc(dosage_matrix, variant_table)
    filtered_variant_table, filtered_raw = filter_variants(
        dosage_matrix,
        variant_qc,
        maf_min=genotype_config.maf_min,
        call_rate_min=genotype_config.call_rate_min,
    )
    sample_out = sample_table.copy()
    sample_out.insert(0, "reference", reference)
    sample_out["call_rate_after_variant_filter"] = 1.0 - np.isnan(filtered_raw).mean(axis=0)
    sample_out["missing_rate_after_variant_filter"] = 1.0 - sample_out["call_rate_after_variant_filter"]
    variant_out = filtered_variant_table.copy()
    variant_out.insert(0, "reference", reference)
    variant_out["missing_rate_after_variant_filter"] = 1.0 - variant_out["call_rate"]
    return sample_out, variant_out


def prepare_knn_imputed_phenotyped_matrix(
    vcf_path: Path,
    phenotype_data,
    genotype_config,
    n_neighbors: int,
) -> np.ndarray:
    """Build a KNN-imputed standardized phenotyped marker matrix, aligned to targets."""
    from cacao_resilience_pipeline.genotypes import (
        build_sample_table,
        compute_variant_qc,
        filter_variants,
        read_vcf_dosage,
        standardize_markers,
    )

    variant_table, sample_names, dosage_matrix = read_vcf_dosage(vcf_path)
    sample_table = build_sample_table(sample_names, phenotype_data.accession_table, phenotype_data.sequencing_summary)
    variant_qc = compute_variant_qc(dosage_matrix, variant_table)
    _, filtered_raw = filter_variants(
        dosage_matrix,
        variant_qc,
        maf_min=genotype_config.maf_min,
        call_rate_min=genotype_config.call_rate_min,
    )
    X_all = filtered_raw.T  # samples x markers
    imputer = KNNImputer(n_neighbors=int(n_neighbors), weights="distance")
    X_all_imputed = imputer.fit_transform(X_all)
    X_all_std = standardize_markers(X_all_imputed)

    sample_table = sample_table.copy()
    sample_table["phenotyped"] = sample_table["sample_norm"].isin(phenotype_data.phenotyped_accessions["sample_norm"])
    phenotyped_table = sample_table.loc[sample_table["phenotyped"]].copy().reset_index(drop=True)
    accession_order = phenotype_data.totals["accession_norm"].tolist()
    lookup = {acc: idx for idx, acc in enumerate(phenotyped_table["sample_norm"].tolist())}
    aligned_indices = [lookup[acc] for acc in accession_order]
    return X_all_std[sample_table["phenotyped"].to_numpy(), :][aligned_indices, :]


# -----------------------------------------------------------------------------
# Analysis runners
# -----------------------------------------------------------------------------


def run_reference_sensitivity_analyses(
    reference: str,
    vcf_path: Path,
    phenotype_data,
    targets: pd.DataFrame,
    genotype_config,
    prediction_config,
    seed: int,
    n_permutations: int,
    block_grid: list[int],
    blocking_methods: list[str],
    run_knn: bool,
    knn_neighbors: int,
    max_targets: int | None,
) -> dict[str, pd.DataFrame]:
    from cacao_resilience_pipeline.genotypes import prepare_reference

    print(f"[{now_stamp()}] Preparing genotype reference: {reference} ({vcf_path.name})")
    genotype_data, variant_qc, sample_qc = prepare_reference(
        reference_name=reference,
        vcf_path=vcf_path,
        phenotype_data=phenotype_data,
        genotype_config=genotype_config,
    )
    accessions = phenotype_data.totals["accession_norm"].tolist()
    X = genotype_data.phenotyped_matrix.astype(float)
    pc_cols = [c for c in genotype_data.pcs_phenotyped.columns if c.startswith("PC")]
    pcs = genotype_data.pcs_phenotyped[pc_cols].to_numpy(dtype=float)

    target_cols = [c for c in PREFERRED_TARGETS if c in targets.columns]
    if max_targets is not None:
        target_cols = target_cols[: int(max_targets)]

    # QC summaries
    print(f"[{now_stamp()}]   QC summaries")
    map_cov, spacing = marker_map_coverage(reference, genotype_data.filtered_variant_table)
    eff_dim = effective_dimension_summary(reference, X)
    sample_missing, variant_missing = filtered_sample_missingness(reference, vcf_path, phenotype_data, genotype_config)

    marker_qc_summary = pd.DataFrame([
        {
            "reference": reference,
            "n_raw_variants_after_multiallelic_exclusion": int(len(variant_qc)),
            "n_retained_snps_after_qc": int(len(genotype_data.filtered_variant_table)),
            "maf_min": float(genotype_config.maf_min),
            "call_rate_min": float(genotype_config.call_rate_min),
            "mean_variant_call_rate_retained": float(genotype_data.filtered_variant_table["call_rate"].mean()),
            "median_variant_call_rate_retained": float(genotype_data.filtered_variant_table["call_rate"].median()),
            "mean_variant_missing_rate_retained": float((1.0 - genotype_data.filtered_variant_table["call_rate"]).mean()),
            "max_variant_missing_rate_retained": float((1.0 - genotype_data.filtered_variant_table["call_rate"]).max()),
            "mean_sample_missing_rate_after_filter": float(sample_missing["missing_rate_after_variant_filter"].mean()),
            "max_sample_missing_rate_after_filter": float(sample_missing["missing_rate_after_variant_filter"].max()),
            "n_phenotyped_accessions_aligned": int(X.shape[0]),
            "n_markers_in_prediction_matrix": int(X.shape[1]),
        }
    ])

    # Alternative blocking and blocked-CV GBLUP summaries
    print(f"[{now_stamp()}]   Alternative blocking + GBLUP blocked CV")
    block_assignment_frames = []
    block_size_frames = []
    alt_rows = []

    for method in blocking_methods:
        for k in block_grid:
            block_ids = make_blocks(X, pcs, method, k, seed)
            block_assignment_frames.append(
                pd.DataFrame({
                    "reference": reference,
                    "block_method": method,
                    "requested_n_blocks": int(k),
                    "actual_n_blocks": int(len(np.unique(block_ids))),
                    "accession_norm": accessions,
                    "block_id": block_ids.astype(int),
                })
            )
            block_size_frames.append(block_summary(reference, method, k, accessions, block_ids))
            for target in target_cols:
                y_raw = targets[target].to_numpy(dtype=float)
                if np.isnan(y_raw).any() or np.nanstd(y_raw) == 0:
                    continue
                y = zscore(y_raw)
                metrics, _, alphas = blocked_cv_gblup(
                    y=y,
                    X_marker=X,
                    block_ids=block_ids,
                    alpha_grid=list(prediction_config.alpha_grid),
                    n_splits_inner=int(prediction_config.n_splits_inner),
                    seed=seed + 1000 * len(method) + k,
                )
                counts = pd.Series(block_ids).value_counts()
                alt_rows.append({
                    "reference": reference,
                    "block_method": method,
                    "requested_n_blocks": int(k),
                    "actual_n_blocks": int(len(np.unique(block_ids))),
                    "min_block_size": int(counts.min()),
                    "max_block_size": int(counts.max()),
                    "target": target,
                    "target_class": target_class(target),
                    "trait_class": disease_class(target),
                    "model": "gblup",
                    **metrics,
                })

    alt_summary = pd.DataFrame(alt_rows)
    if not alt_summary.empty:
        alt_summary["rank_within_reference_blocking"] = (
            alt_summary.groupby(["reference", "block_method", "requested_n_blocks"])["pearson_r"]
            .rank(ascending=False, method="dense")
            .astype(int)
        )

    # Primary high-resolution permutation + FDR on Ward marker, five-block setting.
    # Primary high-resolution empirical support analysis for Table 2 statistics.
    print(f"[{now_stamp()}]   High-resolution permutation tests: n={n_permutations}")
    primary_block_ids = make_blocks(X, pcs, "ward_marker_euclidean", 5, seed)
    perm_rows = []
    for idx, target in enumerate(target_cols, start=1):
        print(f"[{now_stamp()}]     {reference} permutation {idx}/{len(target_cols)}: {target}")
        y_raw = targets[target].to_numpy(dtype=float)
        if np.isnan(y_raw).any() or np.nanstd(y_raw) == 0:
            continue
        y = zscore(y_raw)
        final_metrics, _, alphas = blocked_cv_gblup(
            y=y,
            X_marker=X,
            block_ids=primary_block_ids,
            alpha_grid=list(prediction_config.alpha_grid),
            n_splits_inner=int(prediction_config.n_splits_inner),
            seed=seed + 177 + idx,
        )
        fixed_alpha = float(np.nanmedian(alphas)) if alphas else float(prediction_config.alpha_grid[0])
        pstats = permutation_test_fast(
            y=y,
            X_marker=X,
            block_ids=primary_block_ids,
            alpha=fixed_alpha,
            n_permutations=int(n_permutations),
            seed=seed + 100_000 + idx,
        )
        perm_rows.append({
            "reference": reference,
            "block_method": "ward_marker_euclidean",
            "requested_n_blocks": 5,
            "actual_n_blocks": int(len(np.unique(primary_block_ids))),
            "target": target,
            "target_class": target_class(target),
            "trait_class": disease_class(target),
            "model": "gblup",
            "final_blocked_cv_pearson_r": final_metrics.get("pearson_r", np.nan),
            "final_blocked_cv_spearman_rho": final_metrics.get("spearman_rho", np.nan),
            "final_blocked_cv_rmse": final_metrics.get("rmse", np.nan),
            "final_blocked_cv_mae": final_metrics.get("mae", np.nan),
            "final_blocked_cv_r2": final_metrics.get("r2", np.nan),
            "fixed_alpha_for_permutation": fixed_alpha,
            **pstats,
        })

    perm_summary = pd.DataFrame(perm_rows)

    # Imputation sensitivity: mean imputation from original pipeline vs KNN imputation.
    impute_rows = []
    if run_knn:
        print(f"[{now_stamp()}]   KNN-imputation sensitivity")
        try:
            X_knn = prepare_knn_imputed_phenotyped_matrix(vcf_path, phenotype_data, genotype_config, knn_neighbors)
            for target in target_cols:
                y_raw = targets[target].to_numpy(dtype=float)
                if np.isnan(y_raw).any() or np.nanstd(y_raw) == 0:
                    continue
                y = zscore(y_raw)
                for method_name, X_matrix in [("mean_imputation", X), (f"knn_imputation_k{knn_neighbors}", X_knn)]:
                    metrics, _, _ = blocked_cv_gblup(
                        y=y,
                        X_marker=X_matrix,
                        block_ids=primary_block_ids,  # keep block assignment fixed to isolate imputation effect
                        alpha_grid=list(prediction_config.alpha_grid),
                        n_splits_inner=int(prediction_config.n_splits_inner),
                        seed=seed + 909,
                    )
                    impute_rows.append({
                        "reference": reference,
                        "imputation_method": method_name,
                        "block_method": "ward_marker_euclidean_fixed_from_mean_matrix",
                        "target": target,
                        "target_class": target_class(target),
                        "trait_class": disease_class(target),
                        **metrics,
                    })
        except Exception as exc:
            impute_rows.append({
                "reference": reference,
                "imputation_method": "knn_imputation_failed",
                "error": repr(exc),
            })
    else:
        impute_rows.append({"reference": reference, "imputation_method": "not_run", "note": "--skip-knn was used"})

    return {
        "marker_qc_summary": marker_qc_summary,
        "marker_map_coverage": map_cov,
        "marker_spacing_long": spacing,
        "effective_marker_dimension": eff_dim,
        "sample_missingness_after_filter": sample_missing,
        "variant_missingness_after_filter": variant_missing,
        "block_assignments": pd.concat(block_assignment_frames, ignore_index=True) if block_assignment_frames else pd.DataFrame(),
        "block_size_summary": pd.concat(block_size_frames, ignore_index=True) if block_size_frames else pd.DataFrame(),
        "alternative_blocking_gblup": alt_summary,
        "permutation_fdr_gblup": perm_summary,
        "imputation_sensitivity": pd.DataFrame(impute_rows),
    }


def summarize_static_vs_longitudinal(alt: pd.DataFrame) -> pd.DataFrame:
    if alt.empty:
        return pd.DataFrame()
    # Use the primary five-block setting to avoid overcounting many block variants.
    frame = alt[(alt["block_method"] == "ward_marker_euclidean") & (alt["requested_n_blocks"] == 5)].copy()
    if frame.empty:
        frame = alt.copy()
    return (
        frame.groupby(["target_class", "trait_class"], as_index=False)
        .agg(
            n_tests=("pearson_r", "size"),
            mean_pearson_r=("pearson_r", "mean"),
            median_pearson_r=("pearson_r", "median"),
            min_pearson_r=("pearson_r", "min"),
            max_pearson_r=("pearson_r", "max"),
        )
        .sort_values(["mean_pearson_r"], ascending=False)
        .reset_index(drop=True)
    )


def software_versions() -> pd.DataFrame:
    import scipy
    import sklearn
    try:
        import statsmodels
        statsmodels_version = statsmodels.__version__
    except Exception:
        statsmodels_version = "not_importable"
    try:
        import openpyxl
        openpyxl_version = openpyxl.__version__
    except Exception:
        openpyxl_version = "not_importable"
    rows = [
        ("python", sys.version.replace("\n", " ")),
        ("platform", platform.platform()),
        ("numpy", np.__version__),
        ("pandas", pd.__version__),
        ("scipy", scipy.__version__),
        ("scikit-learn", sklearn.__version__),
        ("statsmodels", statsmodels_version),
        ("openpyxl", openpyxl_version),
    ]
    return pd.DataFrame(rows, columns=["software", "version"])


def write_outputs(output_dir: Path, sheets: dict[str, pd.DataFrame]) -> None:
    ensure_dir(output_dir)
    xlsx_path = output_dir / "longitudinal_prediction_sensitivity_outputs.xlsx"
    print(f"[{now_stamp()}] Writing workbook: {xlsx_path}")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            safe_name = name[:31]
            if frame is None:
                frame = pd.DataFrame()
            # Excel sheet row limit guard. Write a compact summary if accidentally too large.
            if len(frame) > 1_000_000:
                frame.head(1_000_000).to_excel(writer, index=False, sheet_name=safe_name)
            else:
                frame.to_excel(writer, index=False, sheet_name=safe_name)
    # Also write each sheet as CSV for easy inspection/versioning.
    csv_dir = ensure_dir(output_dir / "csv")
    for name, frame in sheets.items():
        if frame is not None and not frame.empty:
            frame.to_csv(csv_dir / f"{name}.csv", index=False)
    print(f"[{now_stamp()}] Done.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sensitivity and reproducibility analyses for cacao longitudinal genomic prediction.")
    parser.add_argument("--root", default=".", help="Project root containing File_S1..File_S6 and pipeline folders.")
    parser.add_argument("--output", default="results_sensitivity", help="Output folder for sensitivity analyses.")
    parser.add_argument("--config", default=None, help="Optional YAML config for the original pipeline.")
    parser.add_argument("--reference", choices=["criollo", "matina", "both"], default="both")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Permutations for blocked-CV empirical tests.")
    parser.add_argument("--block-grid", nargs="+", type=int, default=[4, 5, 6, 7], help="Block counts for alternative blocking sensitivity.")
    parser.add_argument(
        "--blocking-methods",
        nargs="+",
        default=["ward_marker_euclidean", "kmeans_pc", "kmeans_grm_eigenvectors"],
        choices=["ward_marker_euclidean", "kmeans_pc", "kmeans_grm_eigenvectors"],
    )
    parser.add_argument("--skip-knn", action="store_true", help="Skip KNN imputation sensitivity analysis.")
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--max-targets", type=int, default=None, help="Debug option: limit number of targets.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    ensure_dir(output_dir)

    add_project_package_to_path(root)

    from cacao_resilience_pipeline.config import load_config
    from cacao_resilience_pipeline.io import build_file_inventory, discover_input_files
    from cacao_resilience_pipeline.phenotypes import (
        build_prediction_targets,
        fit_trajectory_models,
        load_phenotype_data,
    )

    app_config = load_config(Path(args.config) if args.config else None)
    if args.seed is not None:
        app_config.run.seed = int(args.seed)
    app_config.run.reference = args.reference

    print(f"[{now_stamp()}] Root: {root}")
    discovered = discover_input_files(root, app_config.paths)
    inventory = build_file_inventory(discovered)

    required = ["accession_table", "phenotype_table", "sequencing_table"]
    for key in required:
        if discovered.get(key) is None:
            raise FileNotFoundError(f"Missing required input {key}; discovered={discovered}")

    print(f"[{now_stamp()}] Loading phenotype data and trajectory targets")
    phenotype_data = load_phenotype_data(
        accession_table_path=discovered["accession_table"],
        phenotype_table_path=discovered["phenotype_table"],
        sequencing_table_path=discovered["sequencing_table"],
        phenotype_config=app_config.run.phenotypes,
    )
    fixed_effects, variance_components, accession_features = fit_trajectory_models(phenotype_data.long)
    targets = build_prediction_targets(phenotype_data.totals, accession_features)

    data_summary = pd.DataFrame([
        {"item": "phenotyped_accessions", "value": int(len(phenotype_data.phenotyped_accessions))},
        {"item": "accessions_with_prediction_targets", "value": int(len(targets))},
        {"item": "harvest_periods", "value": int(phenotype_data.long["harvest"].nunique())},
        {"item": "long_format_rows", "value": int(len(phenotype_data.long))},
        {"item": "traits_in_long_panel", "value": "total_pods; healthy_pods; fpr_audpc; flower_wbd_audpc; branch_wbd_audpc"},
        {"item": "target_columns", "value": "; ".join([c for c in PREFERRED_TARGETS if c in targets.columns])},
    ])

    reference_paths = {}
    if args.reference in {"criollo", "both"}:
        if discovered.get("criollo_vcf") is None:
            raise FileNotFoundError("Missing Criollo VCF; expected pattern like *criollo*.vcf")
        reference_paths["criollo"] = discovered["criollo_vcf"]
    if args.reference in {"matina", "both"}:
        if discovered.get("matina_vcf") is None:
            raise FileNotFoundError("Missing Matina VCF; expected pattern like *matina*.vcf")
        reference_paths["matina"] = discovered["matina_vcf"]

    all_sheets: dict[str, list[pd.DataFrame] | pd.DataFrame] = {
        "run_settings": pd.DataFrame([
            {"parameter": "root", "value": str(root)},
            {"parameter": "output", "value": str(output_dir)},
            {"parameter": "seed", "value": str(app_config.run.seed)},
            {"parameter": "reference", "value": str(args.reference)},
            {"parameter": "n_permutations", "value": str(args.n_permutations)},
            {"parameter": "block_grid", "value": ",".join(map(str, args.block_grid))},
            {"parameter": "blocking_methods", "value": ",".join(args.blocking_methods)},
            {"parameter": "skip_knn", "value": str(bool(args.skip_knn))},
            {"parameter": "knn_neighbors", "value": str(args.knn_neighbors)},
            {"parameter": "maf_min", "value": str(app_config.run.genotypes.maf_min)},
            {"parameter": "call_rate_min", "value": str(app_config.run.genotypes.call_rate_min)},
            {"parameter": "n_pcs", "value": str(app_config.run.genotypes.n_pcs)},
            {"parameter": "alpha_grid", "value": json.dumps(list(app_config.run.prediction.alpha_grid))},
        ]),
        "file_inventory": inventory,
        "data_summary": data_summary,
        "trajectory_fixed_effects": fixed_effects,
        "trajectory_variance": variance_components,
        "prediction_targets": targets,
        "software_versions": software_versions(),
        # lists filled below
        "marker_qc_summary": [],
        "marker_map_coverage": [],
        "effective_marker_dimension": [],
        "sample_missingness_after_filter": [],
        "variant_missingness_after_filter": [],
        "block_assignments": [],
        "block_size_summary": [],
        "alternative_blocking_gblup": [],
        "permutation_fdr_gblup": [],
        "imputation_sensitivity": [],
    }

    for ref_name, vcf_path in reference_paths.items():
        ref_results = run_reference_sensitivity_analyses(
            reference=ref_name,
            vcf_path=vcf_path,
            phenotype_data=phenotype_data,
            targets=targets,
            genotype_config=app_config.run.genotypes,
            prediction_config=app_config.run.prediction,
            seed=int(app_config.run.seed),
            n_permutations=int(args.n_permutations),
            block_grid=list(args.block_grid),
            blocking_methods=list(args.blocking_methods),
            run_knn=not bool(args.skip_knn),
            knn_neighbors=int(args.knn_neighbors),
            max_targets=args.max_targets,
        )
        for key, frame in ref_results.items():
            if key in all_sheets and isinstance(all_sheets[key], list):
                all_sheets[key].append(frame)

    # Concatenate list-valued sheets.
    final_sheets: dict[str, pd.DataFrame] = {}
    for key, value in all_sheets.items():
        if isinstance(value, list):
            final_sheets[key] = pd.concat(value, ignore_index=True) if value else pd.DataFrame()
        else:
            final_sheets[key] = value

    # FDR across all reference-target tests in the primary permutation table.
    perm = final_sheets.get("permutation_fdr_gblup", pd.DataFrame()).copy()
    if not perm.empty and "empirical_p_value" in perm.columns:
        perm["bh_fdr_q_value_across_reference_target_tests"] = bh_fdr(perm["empirical_p_value"])
        perm["fdr_0_05"] = perm["bh_fdr_q_value_across_reference_target_tests"] <= 0.05
        perm["fdr_0_10"] = perm["bh_fdr_q_value_across_reference_target_tests"] <= 0.10
        perm = perm.sort_values(["bh_fdr_q_value_across_reference_target_tests", "reference", "target"]).reset_index(drop=True)
        final_sheets["permutation_fdr_gblup"] = perm

    alt = final_sheets.get("alternative_blocking_gblup", pd.DataFrame()).copy()
    final_sheets["static_vs_longitudinal_summary"] = summarize_static_vs_longitudinal(alt)
    if not alt.empty:
        # Stability of target ranks across alternative blocking methods/settings.
        final_sheets["target_rank_stability_blocks"] = (
            alt.groupby(["reference", "target", "trait_class", "target_class"], as_index=False)
            .agg(
                mean_rank=("rank_within_reference_blocking", "mean"),
                sd_rank=("rank_within_reference_blocking", "std"),
                mean_pearson_r=("pearson_r", "mean"),
                min_pearson_r=("pearson_r", "min"),
                max_pearson_r=("pearson_r", "max"),
                n_block_tests=("pearson_r", "size"),
            )
            .sort_values(["reference", "mean_rank", "mean_pearson_r"], ascending=[True, True, False])
            .reset_index(drop=True)
        )

    write_outputs(output_dir, final_sheets)

    print("\nKey output:")
    print(f"  {output_dir / 'longitudinal_prediction_sensitivity_outputs.xlsx'}")
    print("\nNext step after this finishes:")
    print("  Send me the workbook or paste the key sheets: permutation_fdr_gblup, alternative_blocking_gblup,")
    print("  target_rank_stability_blocks, marker_qc_summary, marker_map_coverage, effective_marker_dimension.")


if __name__ == "__main__":
    main()
