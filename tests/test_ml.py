"""Tests for features, training and prediction."""

from __future__ import annotations

import pytest

from michelin.ml.features import TARGET_CLASSES, build_features, build_target, make_feature_spec
from michelin.ml.models import MODEL_REGISTRY, get_model_spec
from michelin.ml.predict import RestaurantInput, predict_restaurant
from michelin.ml.training import TrainingConfig, dataset_fingerprint, train_model


def _config(model_key: str = "decision_tree", **overrides) -> TrainingConfig:
    spec = get_model_spec(model_key)
    params = {**spec.defaults(), **overrides}
    return TrainingConfig(model_key=model_key, params=params)


def test_build_target_groups_stars(training_sample):
    target = build_target(training_sample)
    assert set(target) <= set(TARGET_CLASSES)
    assert (training_sample["Award"].isin(["1 Star", "2 Stars", "3 Stars"]) == (target == "Starred")).all()


def test_build_features_uses_fixed_facility_vocabulary(training_sample):
    spec = make_feature_spec(training_sample)
    X = build_features(training_sample, spec)
    assert list(spec.facility_columns) == [c for c in X.columns if c.startswith("Has: ")]
    assert set(X[list(spec.facility_columns)].stack().unique()) <= {0, 1}


def test_get_model_spec_rejects_unknown_key():
    with pytest.raises(ValueError):
        get_model_spec("svm")


@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY))
def test_every_model_trains(training_sample, model_key):
    result = train_model(
        training_sample, _config(model_key, **({"n_estimators": 50} if model_key == "random_forest" else {}))
    )
    assert 0.0 <= result.accuracy <= 1.0
    assert result.confusion.shape == (len(result.labels), len(result.labels))


def test_hyperparameters_change_the_model(training_sample):
    shallow = train_model(training_sample, _config(max_depth=1))
    deep = train_model(training_sample, _config(max_depth=20, min_samples_leaf=1))
    assert shallow.pipeline.named_steps["model"].get_params()["max_depth"] == 1
    assert deep.pipeline.named_steps["model"].get_params()["max_depth"] == 20


def test_training_rejects_single_class(training_sample):
    single = training_sample[training_sample["Award"] == "Selected Restaurants"].copy()
    with pytest.raises(ValueError, match="única classe"):
        train_model(single, _config())


def test_fingerprint_changes_with_data(training_sample):
    assert dataset_fingerprint(training_sample) == dataset_fingerprint(training_sample.copy())
    assert dataset_fingerprint(training_sample) != dataset_fingerprint(training_sample.iloc[:-1])


def test_prediction_returns_probabilities(training_sample):
    result = train_model(training_sample, _config("logistic_regression"))
    restaurant = RestaurantInput("France", "Creative", 4, ["Interesting wine list", "Valet parking"])
    prediction = predict_restaurant(result, restaurant)

    assert prediction.label in result.labels
    assert prediction.probabilities.sum() == pytest.approx(1.0)
    assert prediction.probabilities.idxmax() == prediction.label


def test_prediction_handles_unseen_country_and_facility(training_sample):
    result = train_model(training_sample, _config())
    restaurant = RestaurantInput("Atlantis", "Moon Food", 2, ["Teleporter"])
    assert predict_restaurant(result, restaurant).label in result.labels
