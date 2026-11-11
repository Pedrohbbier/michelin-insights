"""Clean a raw Michelin table and derive the columns used by charts and models.

The steps follow the routine used in class: inspect nulls and duplicates,
drop what cannot be used, then fix types. Each step records how many rows
it removed so the app can show the user exactly what happened to the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from michelin.config import (
    AWARD_ORDER,
    COL_ADDRESS,
    COL_AWARD,
    COL_CITY,
    COL_COUNTRY,
    COL_CUISINE,
    COL_FACILITIES,
    COL_FACILITY_COUNT,
    COL_FACILITY_LIST,
    COL_GREEN_STAR,
    COL_IS_STARRED,
    COL_LATITUDE,
    COL_LOCATION,
    COL_LONGITUDE,
    COL_NAME,
    COL_PRICE,
    COL_PRICE_LEVEL,
    COUNTRY_ALIASES,
    STAR_AWARDS,
)
from michelin.data.schema import validate_schema

_STAR_PATTERN = re.compile(r"([123])\s*(?:michelin\s*)?stars?", re.IGNORECASE)


@dataclass
class CleaningReport:
    """Summary of what the cleaning step did, shown to the user after upload."""

    rows_in: int
    rows_out: int = 0
    missing_before: dict[str, int] = field(default_factory=dict)
    removed: dict[str, int] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out

    def record(self, reason: str, count: int) -> None:
        if count:
            self.removed[reason] = self.removed.get(reason, 0) + count


def normalize_award(value: object) -> str | None:
    """Map award spellings ("1 MICHELIN Star", "bib gourmand", ...) to a canonical label.

    Returns None when the value is not a known award.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in AWARD_ORDER:
        return text
    lowered = text.lower()
    if "bib" in lowered:
        return "Bib Gourmand"
    if "selected" in lowered:
        return "Selected Restaurants"
    match = _STAR_PATTERN.search(text)
    if match:
        stars = int(match.group(1))
        return "1 Star" if stars == 1 else f"{stars} Stars"
    return None


def price_to_level(value: object) -> float:
    """Convert a price string like "$$$" or "€€" to its level (1-4).

    The guide uses the local currency symbol, so only the number of
    repeated symbols matters. Returns NaN for anything else.
    """
    if not isinstance(value, str):
        return np.nan
    text = value.strip()
    if not text or len(set(text)) != 1 or len(text) > 4:
        return np.nan
    return float(len(text))


def split_location(location: object) -> tuple[str, str]:
    """Split "City, Country" (or "City, State, Country") into (city, country)."""
    if not isinstance(location, str) or not location.strip():
        return ("Unknown", "Unknown")
    parts = [part.strip() for part in location.split(",") if part.strip()]
    city, country = parts[0], parts[-1]
    return city, COUNTRY_ALIASES.get(country, country)


def split_facilities(value: object) -> list[str]:
    """Turn "Air conditioning,Terrace" into ["Air conditioning", "Terrace"]."""
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Columns can mix types (e.g. a JSON file with some numbers written as text),
    # so only actual strings are stripped; other values are kept as they are.
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return df


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate, clean and enrich a raw Michelin table.

    Args:
        raw: DataFrame as read from the uploaded file.

    Returns:
        A tuple with the clean DataFrame and a :class:`CleaningReport`.

    Raises:
        SchemaError: If required columns are missing.
    """
    validate_schema(raw)
    report = CleaningReport(rows_in=len(raw))
    report.missing_before = {col: int(n) for col, n in raw.isnull().sum().items() if n}

    df = _strip_text_columns(raw.copy())

    # 1. Exact duplicates, then the same restaurant listed twice.
    before = len(df)
    df = df.drop_duplicates()
    identity = [COL_NAME, COL_ADDRESS] if COL_ADDRESS in df.columns else [COL_NAME, COL_LOCATION]
    df = df.drop_duplicates(subset=identity, keep="first")
    report.record("Restaurantes duplicados", before - len(df))

    # 2. Award: normalize labels and drop unknown values (it is the ML target).
    df[COL_AWARD] = df[COL_AWARD].map(normalize_award)
    invalid_award = df[COL_AWARD].isna()
    report.record("Prêmio desconhecido ou vazio", int(invalid_award.sum()))
    df = df[~invalid_award]

    # 3. Price: "$$$" -> 3. Rows without a readable price are dropped.
    df[COL_PRICE_LEVEL] = df[COL_PRICE].map(price_to_level)
    invalid_price = df[COL_PRICE_LEVEL].isna()
    report.record("Preço ilegível", int(invalid_price.sum()))
    df = df[~invalid_price]
    df[COL_PRICE_LEVEL] = df[COL_PRICE_LEVEL].astype(int)

    # 4. Coordinates must be numeric and inside valid ranges for the map.
    df[COL_LATITUDE] = pd.to_numeric(df[COL_LATITUDE], errors="coerce")
    df[COL_LONGITUDE] = pd.to_numeric(df[COL_LONGITUDE], errors="coerce")
    invalid_coords = (
        df[COL_LATITUDE].isna()
        | df[COL_LONGITUDE].isna()
        | ~df[COL_LATITUDE].between(-90, 90)
        | ~df[COL_LONGITUDE].between(-180, 180)
    )
    report.record("Coordenadas inválidas", int(invalid_coords.sum()))
    df = df[~invalid_coords]

    # 5. Fill and fix the remaining columns.
    df[COL_CUISINE] = df[COL_CUISINE].fillna("Unknown")
    df[COL_GREEN_STAR] = pd.to_numeric(df[COL_GREEN_STAR], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # 6. Derived columns.
    city_country = df[COL_LOCATION].map(split_location)
    df[COL_CITY] = city_country.str[0]
    df[COL_COUNTRY] = city_country.str[1]
    df[COL_FACILITY_LIST] = df[COL_FACILITIES].map(split_facilities)
    df[COL_FACILITY_COUNT] = df[COL_FACILITY_LIST].map(len)
    df[COL_IS_STARRED] = df[COL_AWARD].isin(STAR_AWARDS)
    df[COL_AWARD] = pd.Categorical(df[COL_AWARD], categories=AWARD_ORDER, ordered=True)

    df = df.reset_index(drop=True)
    report.rows_out = len(df)
    return df, report
