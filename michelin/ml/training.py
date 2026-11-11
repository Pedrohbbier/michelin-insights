"""Train and evaluate classifiers.

Workflow:
1. Split into train/test with a fixed ``random_state``.
2. Fit on the training set and score accuracy on the test set.
3. Build a confusion matrix for the test predictions.
4. Refit on all rows so predictions use every available example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from michelin.config import COL_COUNTRY, COL_CUISINE, RANDOM_STATE
from michelin.ml.features import TARGET_CLASSES, FeatureSpec, build_features, build_target, make_feature_spec
from michelin.ml.models import get_model_spec

# Categories seen fewer times than this are grouped into one "infrequent" column.
MIN_CATEGORY_FREQUENCY = 10
TEST_SIZE = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    model_key: str
    params: dict[str, Any]

    @property
    def model_spec(self):
        return get_model_spec(self.model_key)


@dataclass
class TrainingResult:
    """Everything produced by one training run."""

    config: TrainingConfig
    pipeline: Pipeline
    feature_spec: FeatureSpec
    labels: tuple[str, ...]
    accuracy: float
    confusion: np.ndarray
    n_train: int
    n_test: int
    data_fingerprint: str
    countries: tuple[str, ...]
    cuisines: tuple[str, ...]
    trained_at: datetime = field(default_factory=datetime.now)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Short hash that changes whenever the rows of the dataset change."""
    hashed = pd.util.hash_pandas_object(df[[COL_COUNTRY, COL_CUISINE]].astype(str), index=False)
    return f"{len(df)}-{int(hashed.sum()) & 0xFFFFFFFF:08x}"


def build_pipeline(spec: FeatureSpec, model: Any) -> Pipeline:
    """Preprocessing + model in one object, so the same steps run at train and predict time."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
                list(spec.categorical_columns),
            ),
            ("num", StandardScaler(), list(spec.numeric_columns)),
            ("bin", "passthrough", list(spec.passthrough_columns)),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _validate_classes(y: pd.Series) -> None:
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError("Os dados têm uma única classe; são necessárias pelo menos duas para treinar.")
    if int(counts.min()) < 2:
        raise ValueError(
            f"A classe '{counts.idxmin()}' tem só {int(counts.min())} restaurante(s). "
            "É preciso pelo menos 2 de cada classe para dividir treino e teste."
        )


def train_model(df: pd.DataFrame, config: TrainingConfig) -> TrainingResult:
    """Train and evaluate the configured model on a clean dataset."""
    spec = make_feature_spec(df)
    X = build_features(df, spec)
    y = build_target(df)
    _validate_classes(y)
    labels = tuple(label for label in TARGET_CLASSES if label in set(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(spec, config.model_spec.build(config.params))
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    final_pipeline = clone(pipeline).fit(X, y)

    return TrainingResult(
        config=config,
        pipeline=final_pipeline,
        feature_spec=spec,
        labels=labels,
        accuracy=float(accuracy_score(y_test, y_pred)),
        confusion=confusion_matrix(y_test, y_pred, labels=list(labels)),
        n_train=len(X_train),
        n_test=len(X_test),
        data_fingerprint=dataset_fingerprint(df),
        countries=tuple(sorted(df[COL_COUNTRY].unique())),
        cuisines=tuple(sorted(df[COL_CUISINE].unique())),
    )
