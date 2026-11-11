"""Shared test fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from michelin.config import DEFAULT_DATASET
from michelin.data.cleaning import clean_dataset
from michelin.data.loader import read_path


@pytest.fixture
def raw_rows() -> pd.DataFrame:
    """A tiny hand-written table covering the tricky cases of the real data."""
    return pd.DataFrame(
        {
            "Name": ["Alpha", "Beta", "Gamma", "Delta", "Alpha", "Epsilon", "Zeta"],
            "Address": ["1 Rue A", "2 Main St", "3 Via C", "4 Calle D", "1 Rue A", "5 Road", "6 Ave"],
            "Location": [
                "Paris, France",
                "San Francisco, CA, USA",
                "Rome, Italy",
                "Dubai",
                "Paris, France",
                "Lisbon, Portugal",
                "Tokyo, Japan",
            ],
            "Price": ["$$$$", " €€ ", "$$$", "฿฿", "$$$$", "$$$$$", "¥"],
            "Cuisine": ["French", "Californian", "Italian", None, "French", "Portuguese", "Sushi"],
            "Longitude": [2.35, -122.4, 12.5, 55.3, 2.35, -9.1, "139.7"],
            "Latitude": [48.85, 37.77, 41.9, 25.2, 48.85, 38.7, 35.7],
            "Award": [
                "3 Stars",
                "Bib Gourmand",
                "1 MICHELIN Star",
                "Selected Restaurants",
                "3 Stars",
                "2 Stars",
                "Plate",
            ],
            "GreenStar": [1, 0, 0, None, 1, 0, 0],
            "FacilitiesAndServices": [
                "Air conditioning,Terrace",
                None,
                "Terrace",
                "Car park, Terrace ",
                "Air conditioning,Terrace",
                "",
                "Counter dining",
            ],
        }
    )


@pytest.fixture(scope="session")
def full_dataset() -> pd.DataFrame:
    df, _ = clean_dataset(read_path(DEFAULT_DATASET))
    return df


@pytest.fixture(scope="session")
def training_sample(full_dataset: pd.DataFrame) -> pd.DataFrame:
    """Stratified sample that keeps every award but trains in seconds."""
    groups = full_dataset.groupby("Award", observed=True)
    samples = [group.sample(min(len(group), 400), random_state=0) for _, group in groups]
    return pd.concat(samples).reset_index(drop=True)
