from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

from .io import list_excel_sheets, read_excel_sheet
from .utils import normalize_accession, zscore


@dataclass
class PhenotypeData:
    accession_table: pd.DataFrame
    phenotyped_accessions: pd.DataFrame
    totals: pd.DataFrame
    long: pd.DataFrame
    sequencing_summary: pd.DataFrame


TRANSFORMED_TRAITS = {
    "healthy_pod_logit": {
        "raw": "healthy_pod_rate",
        "favorable_direction": 1,
    },
    "fpr_audpc_log": {
        "raw": "fpr_audpc",
        "favorable_direction": -1,
    },
    "flower_wbd_audpc_log": {
        "raw": "flower_wbd_audpc",
        "favorable_direction": -1,
    },
    "branch_wbd_audpc_log": {
        "raw": "branch_wbd_audpc",
        "favorable_direction": -1,
    },
    "total_pods_log": {
        "raw": "total_pods",
        "favorable_direction": 1,
    },
}


def _find_header_row(frame: pd.DataFrame, target: str) -> int:
    for idx in range(len(frame)):
        row_values = frame.iloc[idx].astype(str).str.strip().tolist()
        if target in row_values:
            return idx
    raise ValueError(f"Could not find header row containing {target!r}.")


def parse_accession_table(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sheets = list_excel_sheets(path)
    accession_raw = read_excel_sheet(path, sheet_name=sheets[0], header=None)
    header_row = _find_header_row(accession_raw, "Sample")
    accessions = accession_raw.iloc[header_row:, :5].copy().reset_index(drop=True)
    accessions.columns = accessions.iloc[0].tolist()
    accessions = accessions.iloc[1:].copy()
    accessions = accessions.rename(
        columns={
            "Sample": "sample",
            "Source": "source",
            "Reference population": "reference_population",
            "Origin": "origin",
        }
    )
    accessions = accessions[["sample", "source", "reference_population", "origin"]].copy()
    accessions["sample"] = accessions["sample"].astype(str)
    accessions["sample_norm"] = accessions["sample"].map(normalize_accession)
    accessions["reference_population"] = accessions["reference_population"].fillna("-").astype(str)

    def first_non_dash(values):
        valid = [str(value) for value in values if pd.notna(value) and str(value).strip() not in {"", "-", "nan"}]
        return valid[0] if valid else "-"

    aggregated = (
        accessions.groupby("sample_norm", as_index=False)
        .agg(
            sample=("sample", "first"),
            source=("source", "first"),
            reference_population=("reference_population", first_non_dash),
            origin=("origin", "first"),
        )
        .sort_values("sample_norm")
        .reset_index(drop=True)
    )

    phenotyped_sheet = sheets[1] if len(sheets) > 1 else sheets[0]
    phenotyped_raw = read_excel_sheet(path, sheet_name=phenotyped_sheet, header=None)
    phenotyped = pd.DataFrame({"sample_raw": phenotyped_raw.iloc[:, 0].dropna().astype(str).tolist()})
    phenotyped["sample_norm"] = phenotyped["sample_raw"].map(normalize_accession)
    phenotyped = phenotyped.drop_duplicates().reset_index(drop=True)

    return aggregated, phenotyped


def parse_sequencing_summary(path) -> pd.DataFrame:
    raw = read_excel_sheet(path, header=None)
    data = raw.iloc[3:, :9].copy().reset_index(drop=True)
    data.columns = [
        "sample",
        "criollo_total_reads",
        "criollo_mapped_reads",
        "criollo_mapped_percentage",
        "criollo_mean_depth",
        "matina_total_reads",
        "matina_mapped_reads",
        "matina_mapped_percentage",
        "matina_mean_depth",
    ]
    numeric_columns = [column for column in data.columns if column != "sample"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["sample"] = data["sample"].astype(str)
    data["sample_norm"] = data["sample"].map(normalize_accession)
    data = data.dropna(subset=["sample"]).reset_index(drop=True)
    return data


def parse_phenotype_table(path, pseudo_count: float, time_order: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_excel_sheet(path, header=None)
    header_row = _find_header_row(raw, "Accession/Harvest")
    table = raw.iloc[header_row:, :6].copy().reset_index(drop=True)
    table.columns = table.iloc[0].tolist()
    table = table.iloc[1:].copy()
    table.columns = [
        "id_or_harvest",
        "total_pods",
        "healthy_pods",
        "fpr_audpc",
        "flower_wbd_audpc",
        "branch_wbd_audpc",
    ]

    total_rows: list[dict] = []
    long_rows: list[dict] = []
    current_accession = None

    for _, row in table.iterrows():
        token = str(row["id_or_harvest"]).strip()
        values = {
            "total_pods": float(row["total_pods"]),
            "healthy_pods": float(row["healthy_pods"]),
            "fpr_audpc": float(row["fpr_audpc"]),
            "flower_wbd_audpc": float(row["flower_wbd_audpc"]),
            "branch_wbd_audpc": float(row["branch_wbd_audpc"]),
        }
        if token in time_order:
            if current_accession is None:
                raise ValueError("Encountered a harvest row before accession header.")
            long_rows.append(
                {
                    "accession_raw": current_accession,
                    "accession_norm": normalize_accession(current_accession),
                    "harvest": token,
                    "time_index": time_order[token],
                    **values,
                }
            )
        else:
            current_accession = token
            total_rows.append(
                {
                    "accession_raw": current_accession,
                    "accession_norm": normalize_accession(current_accession),
                    **values,
                }
            )

    totals = pd.DataFrame(total_rows)
    long = pd.DataFrame(long_rows)

    for frame in (totals, long):
        frame["healthy_pod_rate"] = frame["healthy_pods"] / frame["total_pods"]
        infected = frame["total_pods"] - frame["healthy_pods"]
        frame["healthy_pod_logit"] = np.log((frame["healthy_pods"] + pseudo_count) / (infected + pseudo_count))
        frame["fpr_audpc_log"] = np.log1p(frame["fpr_audpc"])
        frame["flower_wbd_audpc_log"] = np.log1p(frame["flower_wbd_audpc"])
        frame["branch_wbd_audpc_log"] = np.log1p(frame["branch_wbd_audpc"])
        frame["total_pods_log"] = np.log1p(frame["total_pods"])

    mean_time = long["time_index"].mean()
    std_time = long["time_index"].std(ddof=0)
    long["time_scaled"] = (long["time_index"] - mean_time) / std_time

    return totals, long


def load_phenotype_data(accession_table_path, phenotype_table_path, sequencing_table_path, phenotype_config: object) -> PhenotypeData:
    accession_table, phenotyped_accessions = parse_accession_table(accession_table_path)
    sequencing_summary = parse_sequencing_summary(sequencing_table_path)
    totals, long = parse_phenotype_table(
        phenotype_table_path,
        pseudo_count=phenotype_config.pseudo_count,
        time_order=phenotype_config.time_order,
    )
    return PhenotypeData(
        accession_table=accession_table,
        phenotyped_accessions=phenotyped_accessions,
        totals=totals,
        long=long,
        sequencing_summary=sequencing_summary,
    )


def _fit_random_slope_model(frame: pd.DataFrame, response: str) -> tuple[object, str]:
    exog = sm.add_constant(frame[["time_scaled"]])
    try:
        model = MixedLM(frame[response], exog, groups=frame["accession_norm"], exog_re=exog)
        result = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
        return result, "random_slope"
    except Exception:
        model = MixedLM(frame[response], exog, groups=frame["accession_norm"])
        result = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
        return result, "random_intercept"


def _estimate_repeatability(frame: pd.DataFrame, response: str) -> tuple[float, float, float]:
    exog = sm.add_constant(frame[["time_scaled"]])
    ols_result = sm.OLS(frame[response], exog).fit()
    adjusted = frame[response] - ols_result.predict(exog)
    by_accession = pd.DataFrame({"accession_norm": frame["accession_norm"], "adjusted": adjusted})
    accession_means = by_accession.groupby("accession_norm")["adjusted"].mean()
    within_accession = by_accession.groupby("accession_norm")["adjusted"].var(ddof=1).fillna(0.0)
    accession_variance = float(accession_means.var(ddof=1))
    residual_variance = float(within_accession.mean())
    repeatability = accession_variance / (accession_variance + residual_variance / 4.0) if (accession_variance + residual_variance) > 0 else np.nan
    return accession_variance, residual_variance, repeatability


def fit_trajectory_models(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_rows: list[dict] = []
    variance_rows: list[dict] = []
    accession_rows: list[dict] = []

    for response in TRANSFORMED_TRAITS:
        frame = long[["accession_norm", "harvest", "time_index", "time_scaled", response]].copy()
        frame = frame.rename(columns={response: "y"})

        trajectory_result, fit_type = _fit_random_slope_model(frame.rename(columns={"y": response}), response)

        fe = trajectory_result.fe_params
        intercept_fixed = float(fe.get("const", np.nan))
        slope_fixed = float(fe.get("time_scaled", 0.0))

        random_effects = trajectory_result.random_effects
        predicted = []
        for idx, row in frame.iterrows():
            accession = row["accession_norm"]
            re = random_effects.get(accession, None)
            if re is None:
                random_intercept = 0.0
                random_slope = 0.0
            else:
                if hasattr(re, "index"):
                    random_intercept = float(re.iloc[0])
                    random_slope = float(re.iloc[1]) if len(re) > 1 else 0.0
                else:
                    random_intercept = float(re[0])
                    random_slope = float(re[1]) if len(re) > 1 else 0.0
            pred = intercept_fixed + random_intercept + (slope_fixed + random_slope) * row["time_scaled"]
            predicted.append(pred)
        frame["predicted"] = predicted
        frame["residual"] = frame["y"] - frame["predicted"]

        fixed_rows.append(
            {
                "trait": response,
                "fit_type": fit_type,
                "fixed_intercept": intercept_fixed,
                "fixed_slope": slope_fixed,
                "aic": getattr(trajectory_result, "aic", np.nan),
                "bic": getattr(trajectory_result, "bic", np.nan),
                "converged": bool(getattr(trajectory_result, "converged", True)),
                "slope_pvalue": float(trajectory_result.pvalues.get("time_scaled", np.nan)),
            }
        )

        repeatability_var, residual_var, repeatability = _estimate_repeatability(frame.rename(columns={"y": response}), response)
        variance_rows.append(
            {
                "trait": response,
                "accession_variance": repeatability_var,
                "residual_variance": residual_var,
                "repeatability_four_harvests": repeatability,
            }
        )

        grouped = frame.groupby("accession_norm", as_index=False).agg(
            mean=("y", "mean"),
            residual_sd=("residual", "std"),
            first=("y", lambda values: values.iloc[0]),
            last=("y", lambda values: values.iloc[-1]),
        )
        grouped["residual_sd"] = grouped["residual_sd"].fillna(0.0)
        grouped["delta_last_first"] = grouped["last"] - grouped["first"]

        rows = []
        for _, row in grouped.iterrows():
            accession = row["accession_norm"]
            re = random_effects.get(accession, None)
            if re is None:
                random_intercept = 0.0
                random_slope = 0.0
            else:
                if hasattr(re, "index"):
                    random_intercept = float(re.iloc[0])
                    random_slope = float(re.iloc[1]) if len(re) > 1 else 0.0
                else:
                    random_intercept = float(re[0])
                    random_slope = float(re[1]) if len(re) > 1 else 0.0
            rows.append(
                {
                    "accession_norm": accession,
                    f"{response}_intercept": intercept_fixed + random_intercept,
                    f"{response}_slope": slope_fixed + random_slope,
                    f"{response}_mean": float(row["mean"]),
                    f"{response}_residual_sd": float(row["residual_sd"]),
                    f"{response}_delta_last_first": float(row["delta_last_first"]),
                }
            )
        accession_rows.append(pd.DataFrame(rows))

    accession_features = accession_rows[0]
    for frame in accession_rows[1:]:
        accession_features = accession_features.merge(frame, on="accession_norm", how="outer")

    return pd.DataFrame(fixed_rows), pd.DataFrame(variance_rows), accession_features


def build_prediction_targets(totals: pd.DataFrame, accession_features: pd.DataFrame) -> pd.DataFrame:
    legacy = totals[[
        "accession_norm",
        "healthy_pod_rate",
        "healthy_pod_logit",
        "fpr_audpc_log",
        "flower_wbd_audpc_log",
        "branch_wbd_audpc_log",
        "total_pods_log",
    ]].copy()
    legacy = legacy.rename(
        columns={
            "healthy_pod_rate": "legacy_healthy_pod_rate",
            "healthy_pod_logit": "legacy_healthy_pod_logit",
            "fpr_audpc_log": "legacy_fpr_audpc_log",
            "flower_wbd_audpc_log": "legacy_flower_wbd_audpc_log",
            "branch_wbd_audpc_log": "legacy_branch_wbd_audpc_log",
            "total_pods_log": "legacy_total_pods_log",
        }
    )
    return legacy.merge(accession_features, on="accession_norm", how="left")


def build_selection_index(targets: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    score_table = targets[["accession_norm"]].copy()
    available = {column: weight for column, weight in weights.items() if column in targets.columns}
    if not available:
        raise ValueError("No selection-index columns are present in the target table.")

    total_score = pd.Series(np.zeros(len(targets)), index=targets.index, dtype=float)
    for column, weight in available.items():
        standardized = zscore(targets[column])
        score_table[f"z_{column}"] = standardized
        score_table[f"weighted_{column}"] = standardized * weight
        total_score = total_score + standardized * weight

    score_table["selection_index"] = total_score.values
    score_table["selection_rank"] = score_table["selection_index"].rank(ascending=False, method="dense").astype(int)

    objectives = pd.DataFrame(
        {
            "healthy": targets.get("healthy_pod_logit_intercept", pd.Series(np.nan, index=targets.index)),
            "fpr": -targets.get("fpr_audpc_log_intercept", pd.Series(np.nan, index=targets.index)),
            "flower_wbd": -targets.get("flower_wbd_audpc_log_intercept", pd.Series(np.nan, index=targets.index)),
            "branch_wbd": -targets.get("branch_wbd_audpc_log_intercept", pd.Series(np.nan, index=targets.index)),
        }
    )

    pareto = []
    for idx in objectives.index:
        dominated = False
        a = objectives.loc[idx].values
        for jdx in objectives.index:
            if idx == jdx:
                continue
            b = objectives.loc[jdx].values
            if np.all(b >= a) and np.any(b > a):
                dominated = True
                break
        pareto.append(not dominated)
    score_table["pareto_optimal"] = pareto

    return score_table.sort_values(["selection_rank", "accession_norm"]).reset_index(drop=True)
