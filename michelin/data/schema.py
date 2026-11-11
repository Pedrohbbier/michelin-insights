"""Check that an uploaded table has the columns the app needs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from michelin.config import REQUIRED_COLUMNS


class SchemaError(ValueError):
    """Raised when a DataFrame does not match the expected schema."""


@dataclass(frozen=True)
class SchemaCheck:
    missing: tuple[str, ...]
    extra: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing


def check_schema(df: pd.DataFrame) -> SchemaCheck:
    """Compare the DataFrame columns with :data:`REQUIRED_COLUMNS`."""
    columns = set(df.columns)
    missing = tuple(col for col in REQUIRED_COLUMNS if col not in columns)
    extra = tuple(sorted(columns - set(REQUIRED_COLUMNS)))
    return SchemaCheck(missing=missing, extra=extra)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise :class:`SchemaError` if a required column is missing or the table is empty."""
    if df.empty:
        raise SchemaError("O arquivo não tem nenhuma linha.")
    check = check_schema(df)
    if not check.is_valid:
        raise SchemaError("Faltam colunas obrigatórias: " + ", ".join(check.missing))
