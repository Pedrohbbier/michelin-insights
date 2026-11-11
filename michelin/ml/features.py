"""Turn a clean Michelin table into model inputs (X) and target (y)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from michelin.config import (
    COL_AWARD,
    COL_COUNTRY,
    COL_CUISINE,
    COL_FACILITY_COUNT,
    COL_FACILITY_LIST,
    COL_GREEN_STAR,
    COL_PRICE_LEVEL,
    STAR_AWARDS,
)

CATEGORICAL_FEATURES = (COL_COUNTRY, COL_CUISINE)
NUMERIC_FEATURES = (COL_PRICE_LEVEL, COL_FACILITY_COUNT)
BINARY_FEATURES = (COL_GREEN_STAR,)
FACILITY_PREFIX = "Has: "

# Selected / Bib Gourmand / Starred (1+2+3 Stars grouped).
TARGET_CLASSES = ("Selected Restaurants", "Bib Gourmand", "Starred")


@dataclass(frozen=True)
class FeatureSpec:
    """Feature layout fixed at training time and reused at prediction time.

    The facility vocabulary comes from the training data, so a new upload
    with an unseen facility does not change the model's input columns.
    """

    facilities: tuple[str, ...]

    @property
    def facility_columns(self) -> tuple[str, ...]:
        return tuple(FACILITY_PREFIX + name for name in self.facilities)

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return NUMERIC_FEATURES

    @property
    def passthrough_columns(self) -> tuple[str, ...]:
        return BINARY_FEATURES + self.facility_columns

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return CATEGORICAL_FEATURES


def make_feature_spec(df: pd.DataFrame, min_count: int = 1) -> FeatureSpec:
    """Collect the facilities that appear at least ``min_count`` times."""
    counts = df[COL_FACILITY_LIST].explode().dropna().value_counts()
    facilities = tuple(sorted(counts[counts >= min_count].index))
    return FeatureSpec(facilities=facilities)


def build_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Return the feature matrix X with one 0/1 column per known facility."""
    features = df[list(CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES)].copy()
    facility_sets = df[COL_FACILITY_LIST].map(set)
    for name, column in zip(spec.facilities, spec.facility_columns, strict=True):
        features[column] = facility_sets.map(lambda items, n=name: int(n in items))
    return features


def build_target(df: pd.DataFrame) -> pd.Series:
    """Map each award to Selected, Bib Gourmand or Starred."""
    award = df[COL_AWARD].astype(str)
    return award.where(~award.isin(STAR_AWARDS), "Starred").rename("target")
