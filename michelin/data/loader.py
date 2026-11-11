"""Read tabular data from CSV, JSON or XML into a pandas DataFrame."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from michelin.config import SUPPORTED_EXTENSIONS


class UnsupportedFormatError(ValueError):
    """Raised when a file extension is not one of the supported formats."""


def _extension_of(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise UnsupportedFormatError(f"Tipo de arquivo não suportado '{extension}'. Use um destes: {supported}.")
    return extension


def _read_json(buffer: BinaryIO) -> pd.DataFrame:
    # Accept both a list of records ([{...}, {...}]) and pandas' column layout.
    raw = buffer.read()
    try:
        return pd.read_json(io.BytesIO(raw), orient="records")
    except ValueError:
        return pd.read_json(io.BytesIO(raw))


def read_table(source: BinaryIO, filename: str) -> pd.DataFrame:
    """Read a file-like object, choosing the parser from the file name.

    Args:
        source: Binary stream with the file content (e.g. a Streamlit upload).
        filename: Original file name, used only to detect the format.

    Returns:
        The raw DataFrame, with no cleaning applied.
    """
    extension = _extension_of(filename)
    if extension == ".csv":
        return pd.read_csv(source)
    if extension == ".json":
        return _read_json(source)
    # XML: every <row> element under the root becomes one DataFrame row.
    return pd.read_xml(source, parser="lxml")


def read_path(path: str | Path) -> pd.DataFrame:
    """Read a file from disk. Convenience wrapper around :func:`read_table`."""
    path = Path(path)
    with path.open("rb") as handle:
        return read_table(handle, path.name)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV, JSON or XML based on the path extension."""
    path = Path(path)
    extension = _extension_of(path.name)
    if extension == ".csv":
        df.to_csv(path, index=False)
    elif extension == ".json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        df.to_xml(path, index=False, root_name="restaurants", row_name="restaurant", parser="lxml")
