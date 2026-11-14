from io import BytesIO

import pandas as pd
import pytest

from tools.file_loader import load_csv_file, load_excel_sheet
from tools.data_profiler import generate_data_profile


def make_excel_bytes(rows, sheet_name="Data"):
    buffer = BytesIO()
    pd.DataFrame(rows).to_excel(buffer, index=False, header=False, sheet_name=sheet_name)
    return buffer.getvalue()


def test_excel_loader_detects_headers_below_report_titles():
    raw_rows = [
        ["INSTITUTE NAME", None, None],
        ["Department of IT", None, None],
        ["Project Group List", None, None],
        ["Student Name", "Roll Number", "Project Guide"],
        ["Asha", "101", "Dr Rao"],
        ["Ravi", "102", "Dr Sen"],
    ]
    file_bytes = make_excel_bytes(raw_rows)

    df = load_excel_sheet(file_bytes, "Data")

    assert list(df.columns) == ["Student Name", "Roll Number", "Project Guide"]
    assert len(df) == 2
    assert df.iloc[0]["Student Name"] == "Asha"


def test_excel_loader_handles_blank_and_duplicate_headers():
    raw_rows = [
        ["Name", "Name", None],
        ["Asha", "Ravi", "Extra"],
    ]
    file_bytes = make_excel_bytes(raw_rows)

    df = load_excel_sheet(file_bytes, "Data")

    assert list(df.columns) == ["Name", "Name_2", "Column_3"]
    assert len(df) == 1


def test_csv_loader_detects_semicolon_delimiter_and_encoding():
    data = "Name;City\nAndré;Kolkata\n".encode("utf-8")
    df = load_csv_file(data)

    assert list(df.columns) == ["Name", "City"]
    assert df.iloc[0]["Name"] == "André"


def test_text_only_dataset_profiles_without_numeric_crash():
    df = pd.DataFrame({
        "Name": ["Asha", "Ravi"],
        "Department": ["IT", "CSE"],
    })

    report = generate_data_profile(df)

    assert report["dataset_overview"]["total_rows"] == 2
    assert report["statistical_summary"]["numerical_summary"].empty
    assert not report["statistical_summary"]["categorical_summary"].empty


def test_mixed_dataset_profiles_numeric_categorical_and_datetime():
    df = pd.DataFrame({
        "Revenue": [100, 200],
        "Region": ["East", "West"],
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
    })

    report = generate_data_profile(df)

    assert not report["statistical_summary"]["numerical_summary"].empty
    assert not report["statistical_summary"]["categorical_summary"].empty
    assert not report["statistical_summary"]["datetime_summary"].empty


def test_empty_excel_sheet_returns_clear_error():
    file_bytes = make_excel_bytes([[None, None], [None, None]])
    with pytest.raises(ValueError, match="no usable cells"):
        load_excel_sheet(file_bytes, "Data")
