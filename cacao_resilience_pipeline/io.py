from __future__ import annotations

from pathlib import Path
from typing import Iterable
import shutil
import subprocess
import tempfile

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

from .utils import safe_sheet_name


def discover_first(root_path: Path, patterns: Iterable[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(root_path.glob(pattern)))
    return matches[0] if matches else None


def discover_input_files(root_path: Path, path_config) -> dict[str, Path | None]:
    return {
        "accession_table": discover_first(root_path, path_config.accession_table_patterns),
        "phenotype_table": discover_first(root_path, path_config.phenotype_table_patterns),
        "sequencing_table": discover_first(root_path, path_config.sequencing_table_patterns),
        "selection_table": discover_first(root_path, path_config.selection_table_patterns),
        "criollo_vcf": discover_first(root_path, path_config.criollo_vcf_patterns),
        "matina_vcf": discover_first(root_path, path_config.matina_vcf_patterns),
        "tpg_supp_xlsx": discover_first(root_path, path_config.tpg_supp_xlsx_patterns),
        "tpg_supp_docx": discover_first(root_path, path_config.tpg_supp_docx_patterns),
    }


def build_file_inventory(paths: dict[str, Path | None]) -> pd.DataFrame:
    rows = []
    for key, path in paths.items():
        rows.append(
            {
                "input_key": key,
                "exists": path is not None,
                "path": str(path) if path is not None else "",
            }
        )
    return pd.DataFrame(rows)


def _convert_xls_to_xlsx(path: Path) -> Path | None:
    candidates = [shutil.which("libreoffice"), shutil.which("soffice")]
    binary = next((candidate for candidate in candidates if candidate), None)
    if binary is None:
        return None
    temp_root = Path(tempfile.mkdtemp(prefix="cacao_xls_"))
    command = [binary, "--headless", "--convert-to", "xlsx", "--outdir", str(temp_root), str(path)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    converted = temp_root / f"{path.stem}.xlsx"
    return converted if converted.exists() else None


def _open_excel_file(path: Path) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(path)
    except ImportError:
        if path.suffix.lower() != ".xls":
            raise
        converted = _convert_xls_to_xlsx(path)
        if converted is None:
            raise
        return pd.ExcelFile(converted)


def read_excel_sheet(path: Path, sheet_name=0, header=None) -> pd.DataFrame:
    workbook = _open_excel_file(path)
    return workbook.parse(sheet_name=sheet_name, header=header)


def list_excel_sheets(path: Path) -> list[str]:
    workbook = _open_excel_file(path)
    return workbook.sheet_names


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    existing: set[str] = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            safe_name = safe_sheet_name(sheet_name, existing)
            frame.to_excel(writer, sheet_name=safe_name, index=False)

    workbook = load_workbook(path)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter
            for cell in column_cells[:200]:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))
            worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 40)
    workbook.save(path)
