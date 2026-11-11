"""Predict the award of a new (hypothetical) restaurant."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from michelin.config import (
    COL_COUNTRY,
    COL_CUISINE,
    COL_FACILITY_COUNT,
    COL_FACILITY_LIST,
    COL_GREEN_STAR,
    COL_PRICE_LEVEL,
)
from michelin.ml.features import build_features
from michelin.ml.training import TrainingResult


@dataclass(frozen=True)
class RestaurantInput:
    country: str
    cuisine: str
    price_level: int
    facilities: Sequence[str] = ()
    green_star: bool = False

    def to_frame(self) -> pd.DataFrame:
        """One-row DataFrame shaped like a clean dataset row."""
        facilities = list(self.facilities)
        return pd.DataFrame(
            {
                COL_COUNTRY: [self.country],
                COL_CUISINE: [self.cuisine],
                COL_PRICE_LEVEL: [int(self.price_level)],
                COL_FACILITY_COUNT: [len(facilities)],
                COL_GREEN_STAR: [int(self.green_star)],
                COL_FACILITY_LIST: [facilities],
            }
        )


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: pd.Series  # Indexed by class, in the model's class order.


def predict_restaurant(result: TrainingResult, restaurant: RestaurantInput) -> Prediction:
    """Run the trained pipeline on a single restaurant."""
    X = build_features(restaurant.to_frame(), result.feature_spec)
    pipeline = result.pipeline
    label = str(pipeline.predict(X)[0])
    probabilities = pd.Series(pipeline.predict_proba(X)[0], index=pipeline.classes_.astype(str))
    # Show classes in their natural order (Selected -> 3 Stars).
    ordered = [c for c in result.labels if c in probabilities.index]
    return Prediction(label=label, probabilities=probabilities.reindex(ordered))
