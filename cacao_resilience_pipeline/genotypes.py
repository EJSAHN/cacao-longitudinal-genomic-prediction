from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from .utils import extract_accession_from_vcf_sample


@dataclass
class GenotypeData:
    reference_name: str
    variant_table: pd.DataFrame
    sample_table: pd.DataFrame
    dosage_matrix: np.ndarray
    filtered_variant_table: pd.DataFrame
    filtered_dosage_matrix: np.ndarray
    phenotyped_table: pd.DataFrame
    phenotyped_matrix: np.ndarray
    pcs_all: pd.DataFrame
    pcs_phenotyped: pd.DataFrame


def read_vcf_dosage(path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    sample_names: list[str] | None = None
    variants: list[dict] = []
    rows: list[list[float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                sample_names = line.rstrip("\n").split("\t")[9:]
                continue
            fields = line.rstrip("\n").split("\t")
            fmt = fields[8].split(":")
            gt_index = fmt.index("GT") if "GT" in fmt else 0
            genotype_row: list[float] = []
            skip_variant = False
            for value in fields[9:]:
                gt = value.split(":")[gt_index]
                if gt in {"./.", ".", ".|."}:
                    genotype_row.append(np.nan)
                    continue
                separator = "/" if "/" in gt else "|"
                alleles = gt.split(separator)
                try:
                    integers = [int(allele) for allele in alleles]
                except ValueError:
                    genotype_row.append(np.nan)
                    continue
                if max(integers) > 1:
                    skip_variant = True
                    break
                genotype_row.append(float(sum(integers)))
            if skip_variant:
                continue
            variants.append(
                {
                    "chrom": fields[0],
                    "pos": int(fields[1]),
                    "variant_id": f"{fields[0]}_{fields[1]}",
                    "ref": fields[3],
                    "alt": fields[4],
                }
            )
            rows.append(genotype_row)

    if sample_names is None:
        raise ValueError(f"Missing header line in VCF: {path}")

    variant_table = pd.DataFrame(variants)
    dosage_matrix = np.asarray(rows, dtype=float)
    return variant_table, sample_names, dosage_matrix


def build_sample_table(sample_names: list[str], accession_table: pd.DataFrame, sequencing_summary: pd.DataFrame) -> pd.DataFrame:
    sample_table = pd.DataFrame({"vcf_sample": sample_names})
    sample_table["sample_norm"] = sample_table["vcf_sample"].map(extract_accession_from_vcf_sample)
    sample_table = sample_table.merge(accession_table, on="sample_norm", how="left")
    sample_table = sample_table.merge(
        sequencing_summary,
        left_on="sample_norm",
        right_on="sample_norm",
        how="left",
        suffixes=("", "_seq"),
    )
    return sample_table


def compute_variant_qc(dosage_matrix: np.ndarray, variant_table: pd.DataFrame) -> pd.DataFrame:
    call_rate = 1.0 - np.isnan(dosage_matrix).mean(axis=1)
    allele_frequency = np.nanmean(dosage_matrix / 2.0, axis=1)
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    heterozygosity = np.nanmean(dosage_matrix == 1.0, axis=1)
    qc = variant_table.copy()
    qc["call_rate"] = call_rate
    qc["allele_frequency"] = allele_frequency
    qc["maf"] = maf
    qc["heterozygosity"] = heterozygosity
    return qc


def compute_sample_qc(dosage_matrix: np.ndarray, sample_table: pd.DataFrame) -> pd.DataFrame:
    qc = sample_table.copy()
    qc["call_rate"] = 1.0 - np.isnan(dosage_matrix).mean(axis=0)
    qc["heterozygosity"] = np.nanmean(dosage_matrix == 1.0, axis=0)
    return qc


def filter_variants(dosage_matrix: np.ndarray, variant_qc: pd.DataFrame, maf_min: float, call_rate_min: float) -> tuple[pd.DataFrame, np.ndarray]:
    keep = (variant_qc["maf"] >= maf_min) & (variant_qc["call_rate"] >= call_rate_min)
    filtered_table = variant_qc.loc[keep].reset_index(drop=True)
    filtered_matrix = dosage_matrix[keep.to_numpy(), :]
    return filtered_table, filtered_matrix


def _mean_impute(matrix: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    if not np.isnan(filled).any():
        return filled
    means = np.nanmean(filled, axis=1)
    missing = np.where(np.isnan(filled))
    filled[missing] = means[missing[0]]
    return filled


def standardize_markers(samples_by_markers: np.ndarray) -> np.ndarray:
    means = samples_by_markers.mean(axis=0)
    stds = samples_by_markers.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    return (samples_by_markers - means) / stds


def compute_pcs(samples_by_markers: np.ndarray, accession_order: list[str], n_pcs: int) -> pd.DataFrame:
    n_components = min(n_pcs, samples_by_markers.shape[0], samples_by_markers.shape[1])
    pca = PCA(n_components=n_components, random_state=1)
    scores = pca.fit_transform(samples_by_markers)
    frame = pd.DataFrame({"accession_norm": accession_order})
    for index in range(n_components):
        frame[f"PC{index + 1}"] = scores[:, index]
    frame["explained_variance_ratio"] = np.nan
    return frame


def build_relatedness_blocks(samples_by_markers: np.ndarray, accession_order: list[str], n_blocks: int) -> pd.DataFrame:
    effective_blocks = max(2, min(n_blocks, len(accession_order) // 5))
    if effective_blocks >= len(accession_order):
        effective_blocks = max(2, len(accession_order) // 2)
    distances = pdist(samples_by_markers, metric="euclidean")
    linkage_matrix = linkage(distances, method="ward")
    labels = fcluster(linkage_matrix, effective_blocks, criterion="maxclust")
    return pd.DataFrame({"accession_norm": accession_order, "block_id": labels.astype(int)})


def prepare_reference(
    reference_name: str,
    vcf_path: Path,
    phenotype_data,
    genotype_config,
) -> tuple[GenotypeData, pd.DataFrame, pd.DataFrame]:
    variant_table, sample_names, dosage_matrix = read_vcf_dosage(vcf_path)
    sample_table = build_sample_table(sample_names, phenotype_data.accession_table, phenotype_data.sequencing_summary)

    variant_qc = compute_variant_qc(dosage_matrix, variant_table)
    sample_qc = compute_sample_qc(dosage_matrix, sample_table)
    filtered_variant_table, filtered_dosage_matrix = filter_variants(
        dosage_matrix,
        variant_qc,
        maf_min=genotype_config.maf_min,
        call_rate_min=genotype_config.call_rate_min,
    )
    filtered_dosage_matrix = _mean_impute(filtered_dosage_matrix)

    sample_table = sample_qc.copy()
    sample_table["phenotyped"] = sample_table["sample_norm"].isin(phenotype_data.phenotyped_accessions["sample_norm"])

    all_accession_order = sample_table["sample_norm"].tolist()
    all_matrix = filtered_dosage_matrix.T
    all_matrix_standardized = standardize_markers(all_matrix)
    pcs_all = compute_pcs(all_matrix_standardized, all_accession_order, genotype_config.n_pcs)

    phenotyped_table = sample_table.loc[sample_table["phenotyped"]].copy().reset_index(drop=True)
    accession_order = phenotype_data.totals["accession_norm"].tolist()
    phenotyped_lookup = {accession: idx for idx, accession in enumerate(phenotyped_table["sample_norm"].tolist())}
    aligned_indices = [phenotyped_lookup[accession] for accession in accession_order]
    phenotyped_table = phenotyped_table.set_index("sample_norm").loc[accession_order].reset_index()
    phenotyped_matrix = all_matrix_standardized[sample_table["phenotyped"].to_numpy(), :][aligned_indices, :]

    pcs_phenotyped = compute_pcs(phenotyped_matrix, accession_order, genotype_config.n_pcs)
    blocks = build_relatedness_blocks(phenotyped_matrix, accession_order, genotype_config.n_blocks)
    pcs_phenotyped = pcs_phenotyped.merge(blocks, on="accession_norm", how="left")

    genotype_data = GenotypeData(
        reference_name=reference_name,
        variant_table=variant_qc,
        sample_table=sample_qc,
        dosage_matrix=dosage_matrix,
        filtered_variant_table=filtered_variant_table,
        filtered_dosage_matrix=filtered_dosage_matrix,
        phenotyped_table=phenotyped_table,
        phenotyped_matrix=phenotyped_matrix,
        pcs_all=pcs_all,
        pcs_phenotyped=pcs_phenotyped,
    )
    return genotype_data, variant_qc, sample_qc
