# Longitudinal genomic prediction analyses for cacao disease traits

This package contains analysis code for repeated-harvest cacao disease and productivity phenotypes. It reconstructs longitudinal phenotype targets, processes reference-specific SNP marker matrices, runs genomic prediction, and performs sensitivity analyses for blocked validation, empirical permutation testing, marker coverage, effective marker dimensionality, and imputation.

## Expected input files

Place the following files in one project folder before running the scripts:

- `File_S1.xls` or `File_S1.xlsx`
- `File_S2.xls` or `File_S2.xlsx`
- `File_S3.xls` or `File_S3.xlsx`
- `File_S4.xlsx`
- `File_S5_criollo.vcf`
- `File_S6_matina.vcf`

These files correspond to the public supplementary phenotype, metadata, sequencing-summary, and VCF resources from the source cacao diversity-panel study.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate cacao-resilience
```

Using pip:

```bash
python -m pip install -r requirements.txt
```

## Main analysis

```bash
python run_pipeline.py --root /path/to/project_folder --output /path/to/project_folder/results --reference both
```

## Sensitivity and reproducibility analyses

```bash
python scripts/run_prediction_sensitivity.py --root /path/to/project_folder --output /path/to/project_folder/results_sensitivity --reference both --n-permutations 10000
```

The sensitivity script writes a workbook named:

```text
longitudinal_prediction_sensitivity_outputs.xlsx
```

The workbook includes high-resolution empirical permutation results, Benjamini-Hochberg false-discovery-rate summaries, alternative relatedness-blocking analyses, marker-map coverage summaries, effective marker-dimensionality summaries, missingness summaries, imputation sensitivity, and software versions.

## Notes

- Healthy pod rate is transformed using an empirical logit with a pseudocount.
- Disease and pod-count variables are transformed using `log1p`.
- Relatedness-aware validation uses genotype-derived blocks.
- Alternative block definitions include Ward clustering on marker distances, k-means clustering on marker principal components, and k-means clustering on relationship-matrix eigenvectors.
- Random procedures use a fixed seed from the configuration unless overridden.
