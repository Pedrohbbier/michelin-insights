"""Tests for loading, schema validation and cleaning."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from michelin.data.cleaning import clean_dataset, normalize_award, price_to_level, split_facilities, split_location
from michelin.data.loader import UnsupportedFormatError, read_path, read_table, write_table
from michelin.data.schema import SchemaError, check_schema, validate_schema

# --- Loader -------------------------------------------------------------------


@pytest.mark.parametrize("extension", [".csv", ".json", ".xml"])
def test_write_then_read_round_trip(tmp_path, raw_rows, extension):
    path = tmp_path / f"data{extension}"
    write_table(raw_rows.astype(str), path)
    loaded = read_path(path)
    assert list(loaded.columns) == list(raw_rows.columns)
    assert len(loaded) == len(raw_rows)


def test_read_table_rejects_unknown_extension():
    with pytest.raises(UnsupportedFormatError):
        read_table(io.BytesIO(b"a,b"), "data.xlsx")


def test_read_json_accepts_records_layout():
    payload = b'[{"Name": "A", "Price": "$"}, {"Name": "B", "Price": "$$"}]'
    df = read_table(io.BytesIO(payload), "data.JSON")
    assert df["Name"].tolist() == ["A", "B"]


# --- Schema -------------------------------------------------------------------


def test_schema_reports_missing_columns(raw_rows):
    check = check_schema(raw_rows.drop(columns=["Award", "Price"]))
    assert not check.is_valid
    assert set(check.missing) == {"Award", "Price"}


def test_validate_schema_rejects_empty_table(raw_rows):
    with pytest.raises(SchemaError):
        validate_schema(raw_rows.iloc[0:0])


# --- Cleaning helpers -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3 Stars", "3 Stars"),
        ("1 MICHELIN Star", "1 Star"),
        ("2 michelin stars", "2 Stars"),
        ("bib gourmand", "Bib Gourmand"),
        ("Selected Restaurants", "Selected Restaurants"),
        ("Plate", None),
        (None, None),
    ],
)
def test_normalize_award(raw, expected):
    assert normalize_award(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [("$", 1), ("€€€", 3), (" ¥¥ ", 2), ("$$$$", 4)])
def test_price_to_level(raw, expected):
    assert price_to_level(raw) == expected


@pytest.mark.parametrize("raw", ["$$$$$", "$€", "", None, "cheap"])
def test_price_to_level_invalid(raw):
    assert np.isnan(price_to_level(raw))


def test_split_location_handles_states_and_aliases():
    assert split_location("San Francisco, CA, USA") == ("San Francisco", "USA")
    assert split_location("Dubai") == ("Dubai", "United Arab Emirates")
    assert split_location(None) == ("Unknown", "Unknown")


def test_split_facilities():
    assert split_facilities("Car park, Terrace ") == ["Car park", "Terrace"]
    assert split_facilities(None) == []


# --- Full cleaning ---------------------------------------------------------------


def test_clean_dataset_removes_bad_rows_and_reports_why(raw_rows):
    df, report = clean_dataset(raw_rows)

    assert report.rows_in == 7
    assert report.removed == {
        "Restaurantes duplicados": 1,  # second "Alpha"
        "Prêmio desconhecido ou vazio": 1,  # "Plate"
        "Preço ilegível": 1,  # "$$$$$"
    }
    assert report.rows_out == len(df) == 4
    assert report.rows_removed == 3


def test_clean_dataset_derives_columns(raw_rows):
    df, _ = clean_dataset(raw_rows)
    row = df.set_index("Name")

    assert row.loc["Beta", "PriceLevel"] == 2
    assert row.loc["Beta", "Country"] == "USA"
    assert row.loc["Delta", "Country"] == "United Arab Emirates"
    assert row.loc["Delta", "Cuisine"] == "Unknown"
    assert row.loc["Delta", "GreenStar"] == 0
    assert row.loc["Gamma", "Award"] == "1 Star"
    assert row.loc["Alpha", "FacilityList"] == ["Air conditioning", "Terrace"]
    assert row.loc["Beta", "FacilityCount"] == 0
    assert bool(row.loc["Alpha", "IsStarred"]) and not bool(row.loc["Beta", "IsStarred"])
    assert isinstance(df["Award"].dtype, pd.CategoricalDtype) and df["Award"].cat.ordered


def test_clean_dataset_requires_schema(raw_rows):
    with pytest.raises(SchemaError):
        clean_dataset(raw_rows.drop(columns=["Latitude"]))


def test_bundled_dataset_is_clean(full_dataset):
    assert len(full_dataset) > 15_000
    assert full_dataset["PriceLevel"].between(1, 4).all()
    assert not full_dataset.duplicated(subset=["Name", "Address"]).any()
