"""Defensive loaders for common tabular file formats."""

from io import BytesIO
from pathlib import Path
import re

import pandas as pd


def _clean_headers(values):
    headers = []
    used = {}
    for index, value in enumerate(values):
        if pd.isna(value) or not str(value).strip():
            name = f"Column_{index + 1}"
        else:
            name = re.sub(r"\s+", " ", str(value)).strip()
        count = used.get(name, 0) + 1
        used[name] = count
        if count > 1:
            name = f"{name}_{count}"
        headers.append(name)
    return headers


def detect_header_row(raw_df, scan_rows=30):
    """Find a likely header row below introductory report titles."""
    if raw_df.empty:
        return 0
    best_row, best_score = 0, -1
    for row_index in range(min(len(raw_df), scan_rows)):
        values = [
            value for value in raw_df.iloc[row_index].tolist()
            if not pd.isna(value) and str(value).strip()
        ]
        if not values:
            continue
        normalized = [str(value).strip().casefold() for value in values]
        unique_ratio = len(set(normalized)) / len(values)
        text_cells = sum(not isinstance(value, (int, float, bool)) for value in values)
        score = len(values) * 3 + unique_ratio + text_cells / len(values)
        if score > best_score:
            best_score, best_row = score, row_index
    return best_row


def load_excel_sheet(file_bytes, sheet_name):
    """Load an Excel worksheet and infer a header row from its structure."""
    if not file_bytes:
        raise ValueError("The uploaded Excel file is empty.")
    try:
        raw = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=None)
    except Exception as error:
        raise ValueError(f"Unable to read worksheet '{sheet_name}': {error}") from error

    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    if raw.empty or raw.shape[1] == 0:
        raise ValueError(f"Worksheet '{sheet_name}' contains no usable cells.")

    header_row = detect_header_row(raw)
    headers = _clean_headers(raw.iloc[header_row].tolist())
    data = raw.iloc[header_row + 1:].copy()
    data.columns = headers
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if data.empty or data.shape[1] == 0:
        raise ValueError(
            f"No data rows were found below the detected header in worksheet '{sheet_name}'."
        )
    return data


def load_csv_file(file_bytes):
    """Read CSV using common encodings and delimiter inference."""
    if not file_bytes:
        raise ValueError("The uploaded CSV file is empty.")
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(BytesIO(file_bytes), encoding=encoding, sep=None, engine="python")
            break
        except Exception as error:
            last_error = error
    else:
        raise ValueError(f"Unable to read the CSV file: {last_error}")

    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    frame.columns = _clean_headers(frame.columns.tolist())
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("The CSV file contains no usable tabular data.")
    return frame


def load_tabular_file(file_bytes, filename, sheet_name=None):
    """Load CSV or Excel bytes into a validated DataFrame."""
    extension = Path(filename).suffix.lower()
    if extension == ".csv":
        return load_csv_file(file_bytes)
    if extension in {".xlsx", ".xlsm", ".xls"}:
        if sheet_name is None:
            workbook = pd.ExcelFile(BytesIO(file_bytes))
            if not workbook.sheet_names:
                raise ValueError("No worksheets were found in this Excel file.")
            sheet_name = workbook.sheet_names[0]
        return load_excel_sheet(file_bytes, sheet_name)
    raise ValueError(f"Unsupported file type '{extension}'. Supported formats are CSV and Excel.")
