from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, KFold

from .utils import percentile_interval


@dataclass
class PredictionBundle:
    summary: pd.DataFrame
    by_repeat: pd.DataFrame
    predictions: pd.DataFrame
    permutation_tests: pd.DataFrame
    final_scores: pd.DataFrame


def _safe_corr(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = np.asarray(y_true, dtype=float)[mask]
    y_pred_clean = np.asarray(y_pred, dtype=float)[mask]
    if len(y_true_clean) < 2:
        return np.nan
    if np.nanstd(y_true_clean) == 0 or np.nanstd(y_pred_clean) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(function(y_true_clean, y_pred_clean).statistic)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "pearson_r": _safe_corr(pearsonr, y_true, y_pred),
        "spearman_rho": _safe_corr(spearmanr, y_true, y_pred),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def _select_alpha_kernel(K: np.ndarray, y: np.ndarray, alpha_grid: list[float], n_splits_inner: int, seed: int) -> float:
    splitter = KFold(n_splits=min(n_splits_inner, len(y)), shuffle=True, random_state=seed)
    best_alpha = alpha_grid[0]
    best_score = -np.inf
    for alpha in alpha_grid:
        predictions = np.full(len(y), np.nan)
        for train_index, test_index in splitter.split(K):
            kernel_train = K[np.ix_(train_index, train_index)]
            kernel_test = K[np.ix_(test_index, train_index)]
            model = KernelRidge(alpha=alpha, kernel="precomputed")
            model.fit(kernel_train, y[train_index])
            predictions[test_index] = model.predict(kernel_test)
        score = _safe_corr(pearsonr, y, predictions)
        score = -np.inf if pd.isna(score) else score
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return float(best_alpha)


def _fit_predict_model(model_name: str, X_marker_train, X_marker_test, X_pc_train, X_pc_test, y_train, alpha, K_train=None, K_test_train=None):
    if model_name == "mean":
        return np.repeat(np.mean(y_train), len(X_marker_test))
    if model_name == "pc_ridge":
        model = Ridge(alpha=alpha)
        model.fit(X_pc_train, y_train)
        return model.predict(X_pc_test)
    if model_name == "rrblup":
        model = Ridge(alpha=alpha)
        model.fit(X_marker_train, y_train)
        return model.predict(X_marker_test)
    if model_name == "gblup":
        model = KernelRidge(alpha=alpha, kernel="precomputed")
        model.fit(K_train, y_train)
        return model.predict(K_test_train)
    raise ValueError(f"Unknown model: {model_name}")


def _evaluate_scheme(
    scheme_name: str,
    model_name: str,
    y: np.ndarray,
    accessions: list[str],
    X_marker: np.ndarray,
    X_pc: np.ndarray,
    block_ids: np.ndarray,
    prediction_config,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[float]]:
    repeats: list[dict] = []
    predictions: list[dict] = []
    chosen_alphas: list[float] = []
    kernel = (X_marker @ X_marker.T) / X_marker.shape[1]

    if scheme_name == "random_cv":
        splitters = [
            KFold(n_splits=prediction_config.n_splits_random, shuffle=True, random_state=seed + repeat)
            for repeat in range(prediction_config.n_repeats_random)
        ]
    elif scheme_name == "block_cv":
        unique_blocks = np.unique(block_ids)
        splitters = [GroupKFold(n_splits=len(unique_blocks))]
    else:
        raise ValueError(f"Unknown scheme: {scheme_name}")

    for repeat_id, splitter in enumerate(splitters, start=1):
        oof = np.full(len(y), np.nan)
        fold_rows: list[dict] = []
        split_iter = splitter.split(X_marker, y, block_ids) if scheme_name == "block_cv" else splitter.split(X_marker, y)
        for fold_id, (train_index, test_index) in enumerate(split_iter, start=1):
            if model_name == "mean":
                alpha = np.nan
                pred = np.repeat(np.mean(y[train_index]), len(test_index))
            elif model_name == "pc_ridge":
                model = RidgeCV(alphas=prediction_config.alpha_grid)
                model.fit(X_pc[train_index], y[train_index])
                alpha = float(model.alpha_)
                pred = model.predict(X_pc[test_index])
            elif model_name == "rrblup":
                model = RidgeCV(alphas=prediction_config.alpha_grid)
                model.fit(X_marker[train_index], y[train_index])
                alpha = float(model.alpha_)
                pred = model.predict(X_marker[test_index])
            elif model_name == "gblup":
                alpha = _select_alpha_kernel(
                    kernel[np.ix_(train_index, train_index)],
                    y[train_index],
                    prediction_config.alpha_grid,
                    prediction_config.n_splits_inner,
                    seed + repeat_id * 100 + fold_id,
                )
                pred = _fit_predict_model(
                    model_name,
                    X_marker[train_index],
                    X_marker[test_index],
                    X_pc[train_index],
                    X_pc[test_index],
                    y[train_index],
                    alpha,
                    K_train=kernel[np.ix_(train_index, train_index)],
                    K_test_train=kernel[np.ix_(test_index, train_index)],
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            chosen_alphas.append(alpha)
            oof[test_index] = pred
            fold_metrics = compute_metrics(y[test_index], pred)
            fold_rows.append({
                "scheme": scheme_name,
                "model": model_name,
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "alpha": alpha,
                **fold_metrics,
            })
            for local_index, accession in zip(test_index, np.asarray(accessions)[test_index]):
                predictions.append(
                    {
                        "scheme": scheme_name,
                        "model": model_name,
                        "repeat_id": repeat_id,
                        "fold_id": fold_id,
                        "accession_norm": accession,
                        "observed": float(y[local_index]),
                        "predicted": float(oof[local_index]),
                    }
                )

        repeat_metrics = compute_metrics(y, oof)
        alpha_values = [row["alpha"] for row in fold_rows if pd.notna(row["alpha"])]
        repeats.append(
            {
                "scheme": scheme_name,
                "model": model_name,
                "repeat_id": repeat_id,
                "alpha_mean": float(np.mean(alpha_values)) if alpha_values else np.nan,
                **repeat_metrics,
            }
        )

    by_repeat = pd.DataFrame(repeats)
    fold_table = pd.DataFrame(predictions)
    summary_rows = []
    for metric in ["pearson_r", "spearman_rho", "rmse", "mae", "r2"]:
        lower, upper = percentile_interval(by_repeat[metric].dropna().tolist())
        summary_rows.append((metric, float(by_repeat[metric].mean()), float(by_repeat[metric].std(ddof=0)), lower, upper))
    summary = pd.DataFrame(summary_rows, columns=["metric", "mean", "sd", "ci_lower", "ci_upper"])
    summary.insert(0, "model", model_name)
    summary.insert(0, "scheme", scheme_name)
    return summary, by_repeat, fold_table, chosen_alphas


def _permutation_test_gblup(
    scheme_name: str,
    y: np.ndarray,
    X_marker: np.ndarray,
    X_pc: np.ndarray,
    accessions: list[str],
    block_ids: np.ndarray,
    prediction_config,
    seed: int,
    fixed_alpha: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def evaluate_permuted(y_perm: np.ndarray) -> float:
        if scheme_name == "random_cv":
            splitter = KFold(n_splits=prediction_config.n_splits_random, shuffle=True, random_state=seed)
            split_iter = splitter.split(X_marker, y_perm)
        else:
            splitter = GroupKFold(n_splits=len(np.unique(block_ids)))
            split_iter = splitter.split(X_marker, y_perm, block_ids)
        oof = np.full(len(y_perm), np.nan)
        for train_index, test_index in split_iter:
            kernel = (X_marker @ X_marker.T) / X_marker.shape[1]
            pred = _fit_predict_model(
                "gblup",
                X_marker[train_index],
                X_marker[test_index],
                X_pc[train_index],
                X_pc[test_index],
                y_perm[train_index],
                fixed_alpha,
                K_train=kernel[np.ix_(train_index, train_index)],
                K_test_train=kernel[np.ix_(test_index, train_index)],
            )
            oof[test_index] = pred
        return compute_metrics(y_perm, oof)["pearson_r"]

    observed = evaluate_permuted(y)
    null_values = [evaluate_permuted(rng.permutation(y)) for _ in range(prediction_config.n_permutations)]
    empirical_p = (1.0 + np.sum(np.asarray(null_values) >= observed)) / (1.0 + len(null_values))
    return pd.DataFrame(
        {
            "scheme": [scheme_name],
            "model": ["gblup"],
            "observed_pearson_r": [observed],
            "null_mean": [float(np.mean(null_values))],
            "null_sd": [float(np.std(null_values, ddof=0))],
            "empirical_p_value": [float(empirical_p)],
            "n_permutations": [prediction_config.n_permutations],
        }
    )


def run_prediction(targets: pd.DataFrame, genotype_data, prediction_config, seed: int) -> PredictionBundle:
    merged = genotype_data.pcs_phenotyped.merge(targets, on="accession_norm", how="inner")
    accessions = merged["accession_norm"].tolist()
    block_ids = merged["block_id"].to_numpy()
    pc_columns = [column for column in merged.columns if column.startswith("PC")]
    X_pc = merged[pc_columns].to_numpy(dtype=float)
    X_marker = genotype_data.phenotyped_matrix.astype(float)

    summary_frames = []
    repeat_frames = []
    prediction_frames = []
    permutation_frames = []
    final_rows = []

    preferred_targets = [
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
    target_columns = [column for column in preferred_targets if column in targets.columns]
    if getattr(prediction_config, "max_targets", None) is not None:
        target_columns = target_columns[: int(prediction_config.max_targets)]

    for target_name in target_columns:
        y_raw = merged[target_name].to_numpy(dtype=float)
        if np.isnan(y_raw).any() or np.std(y_raw) == 0:
            continue
        y = (y_raw - np.mean(y_raw)) / np.std(y_raw, ddof=0)
        chosen_alphas_by_scheme: dict[str, float] = {}

        for scheme_name in ["random_cv", "block_cv"]:
            for model_name in ["mean", *prediction_config.models]:
                summary, by_repeat, predictions, chosen_alphas = _evaluate_scheme(
                    scheme_name=scheme_name,
                    model_name=model_name,
                    y=y,
                    accessions=accessions,
                    X_marker=X_marker,
                    X_pc=X_pc,
                    block_ids=block_ids,
                    prediction_config=prediction_config,
                    seed=seed,
                )
                summary.insert(0, "target", target_name)
                by_repeat.insert(0, "target", target_name)
                predictions.insert(0, "target", target_name)
                summary_frames.append(summary)
                repeat_frames.append(by_repeat)
                prediction_frames.append(predictions)
                if model_name == "gblup" and chosen_alphas:
                    chosen_alphas_by_scheme[scheme_name] = float(np.nanmedian(chosen_alphas))

            if scheme_name == "block_cv" and scheme_name in chosen_alphas_by_scheme and prediction_config.n_permutations > 0:
                permutation = _permutation_test_gblup(
                    scheme_name=scheme_name,
                    y=y,
                    X_marker=X_marker,
                    X_pc=X_pc,
                    accessions=accessions,
                    block_ids=block_ids,
                    prediction_config=prediction_config,
                    seed=seed,
                    fixed_alpha=chosen_alphas_by_scheme[scheme_name],
                )
                permutation.insert(0, "target", target_name)
                permutation_frames.append(permutation)

        kernel = (X_marker @ X_marker.T) / X_marker.shape[1]
        best_gblup_alpha = chosen_alphas_by_scheme.get("random_cv", prediction_config.alpha_grid[0])
        model = KernelRidge(alpha=best_gblup_alpha, kernel="precomputed")
        model.fit(kernel, y)
        fitted = model.predict(kernel)
        for accession, observed, fitted_value in zip(accessions, y, fitted):
            final_rows.append(
                {
                    "target": target_name,
                    "reference": genotype_data.reference_name,
                    "final_model": "gblup",
                    "alpha": best_gblup_alpha,
                    "accession_norm": accession,
                    "observed_standardized": float(observed),
                    "fitted_standardized": float(fitted_value),
                }
            )

    return PredictionBundle(
        summary=pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
        by_repeat=pd.concat(repeat_frames, ignore_index=True) if repeat_frames else pd.DataFrame(),
        predictions=pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        permutation_tests=pd.concat(permutation_frames, ignore_index=True) if permutation_frames else pd.DataFrame(),
        final_scores=pd.DataFrame(final_rows),
    )
