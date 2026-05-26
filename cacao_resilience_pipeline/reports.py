from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import write_workbook


def write_run_inventory(output_path: Path, inventory: pd.DataFrame, run_config: dict) -> None:
    config_rows = []
    for top_key, value in run_config.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                config_rows.append({"parameter": f"{top_key}.{nested_key}", "value": str(nested_value)})
        else:
            config_rows.append({"parameter": top_key, "value": str(value)})
    write_workbook(
        output_path,
        {
            "file_inventory": inventory,
            "run_config": pd.DataFrame(config_rows),
        },
    )


def write_phenotype_workbook(
    output_path: Path,
    accession_table: pd.DataFrame,
    phenotyped_accessions: pd.DataFrame,
    sequencing_summary: pd.DataFrame,
    totals: pd.DataFrame,
    long: pd.DataFrame,
    fixed_effects: pd.DataFrame,
    variance_components: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    write_workbook(
        output_path,
        {
            "accessions": accession_table,
            "phenotyped_accessions": phenotyped_accessions,
            "sequencing_summary": sequencing_summary,
            "phenotype_totals": totals,
            "phenotype_long": long,
            "trajectory_fixed_effects": fixed_effects,
            "trajectory_variance": variance_components,
            "prediction_targets": targets,
        },
    )


def write_prediction_workbook(
    output_path: Path,
    variant_qc: pd.DataFrame,
    sample_qc: pd.DataFrame,
    pcs_all: pd.DataFrame,
    pcs_phenotyped: pd.DataFrame,
    prediction_bundle,
) -> None:
    write_workbook(
        output_path,
        {
            "variant_qc": variant_qc,
            "sample_qc": sample_qc,
            "pcs_all": pcs_all,
            "pcs_phenotyped": pcs_phenotyped,
            "prediction_summary": prediction_bundle.summary,
            "prediction_by_repeat": prediction_bundle.by_repeat,
            "prediction_pairs": prediction_bundle.predictions,
            "permutation_tests": prediction_bundle.permutation_tests,
            "final_scores": prediction_bundle.final_scores,
        },
    )


def write_selection_workbook(output_path: Path, selection_index: pd.DataFrame) -> None:
    write_workbook(output_path, {"selection_index": selection_index})


def write_cross_reference_workbook(output_path: Path, bundles: dict[str, object]) -> None:
    summary_frames = []
    final_frames = []
    for reference_name, bundle in bundles.items():
        summary = bundle.summary.copy()
        summary.insert(0, "reference", reference_name)
        summary_frames.append(summary)
        final_scores = bundle.final_scores.copy()
        final_frames.append(final_scores)
    write_workbook(
        output_path,
        {
            "prediction_summary": pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
            "final_scores": pd.concat(final_frames, ignore_index=True) if final_frames else pd.DataFrame(),
        },
    )
