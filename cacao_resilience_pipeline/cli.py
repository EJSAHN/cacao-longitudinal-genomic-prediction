from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .genotypes import prepare_reference
from .io import build_file_inventory, discover_input_files
from .phenotypes import build_prediction_targets, build_selection_index, fit_trajectory_models, load_phenotype_data
from .prediction import run_prediction
from .reports import (
    write_cross_reference_workbook,
    write_phenotype_workbook,
    write_prediction_workbook,
    write_run_inventory,
    write_selection_workbook,
)
from .utils import ensure_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Longitudinal genomic prediction pipeline for cacao.")
    parser.add_argument("--root", required=True, help="Root path containing the input files.")
    parser.add_argument("--output", required=True, help="Output path for result workbooks.")
    parser.add_argument("--config", default=None, help="Optional YAML configuration file.")
    parser.add_argument("--reference", choices=["criollo", "matina", "both"], default=None)
    parser.add_argument("--fast", action="store_true", help="Run a reduced smoke-test configuration.")
    return parser


def _apply_fast_mode(app_config) -> None:
    app_config.run.prediction.n_repeats_random = 2
    app_config.run.prediction.n_permutations = 0
    app_config.run.prediction.alpha_grid = [10.0, 1000.0]
    app_config.run.prediction.max_targets = 4


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    app_config = load_config(Path(args.config) if args.config else None)
    if args.reference is not None:
        app_config.run.reference = args.reference
    if args.fast:
        _apply_fast_mode(app_config)

    root_path = Path(args.root)
    output_path = ensure_path(Path(args.output))
    discovered_paths = discover_input_files(root_path, app_config.paths)
    inventory = build_file_inventory(discovered_paths)

    required = ["accession_table", "phenotype_table", "sequencing_table"]
    for key in required:
        if discovered_paths[key] is None:
            raise FileNotFoundError(f"Missing required input: {key}")

    phenotype_data = load_phenotype_data(
        accession_table_path=discovered_paths["accession_table"],
        phenotype_table_path=discovered_paths["phenotype_table"],
        sequencing_table_path=discovered_paths["sequencing_table"],
        phenotype_config=app_config.run.phenotypes,
    )
    fixed_effects, variance_components, accession_features = fit_trajectory_models(phenotype_data.long)
    targets = build_prediction_targets(phenotype_data.totals, accession_features)
    selection_index = build_selection_index(targets, app_config.run.selection_index.weights)

    write_run_inventory(output_path / "00_run_inventory.xlsx", inventory, asdict(app_config.run))
    write_phenotype_workbook(
        output_path / "01_phenotype_modeling.xlsx",
        accession_table=phenotype_data.accession_table,
        phenotyped_accessions=phenotype_data.phenotyped_accessions,
        sequencing_summary=phenotype_data.sequencing_summary,
        totals=phenotype_data.totals,
        long=phenotype_data.long,
        fixed_effects=fixed_effects,
        variance_components=variance_components,
        targets=targets,
    )
    write_selection_workbook(output_path / "03_selection_index.xlsx", selection_index)

    reference_paths = {}
    if app_config.run.reference in {"criollo", "both"}:
        if discovered_paths["criollo_vcf"] is None:
            raise FileNotFoundError("Missing Criollo VCF.")
        reference_paths["criollo"] = discovered_paths["criollo_vcf"]
    if app_config.run.reference in {"matina", "both"}:
        if discovered_paths["matina_vcf"] is None:
            raise FileNotFoundError("Missing Matina VCF.")
        reference_paths["matina"] = discovered_paths["matina_vcf"]

    prediction_bundles = {}
    for reference_name, vcf_path in reference_paths.items():
        genotype_data, variant_qc, sample_qc = prepare_reference(
            reference_name=reference_name,
            vcf_path=vcf_path,
            phenotype_data=phenotype_data,
            genotype_config=app_config.run.genotypes,
        )
        prediction_bundle = run_prediction(
            targets=targets,
            genotype_data=genotype_data,
            prediction_config=app_config.run.prediction,
            seed=app_config.run.seed,
        )
        prediction_bundles[reference_name] = prediction_bundle
        write_prediction_workbook(
            output_path / f"02_prediction_{reference_name}.xlsx",
            variant_qc=variant_qc,
            sample_qc=sample_qc,
            pcs_all=genotype_data.pcs_all,
            pcs_phenotyped=genotype_data.pcs_phenotyped,
            prediction_bundle=prediction_bundle,
        )

    if len(prediction_bundles) > 1:
        write_cross_reference_workbook(output_path / "04_cross_reference_summary.xlsx", prediction_bundles)
