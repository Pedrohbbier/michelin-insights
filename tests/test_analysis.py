"""Tests for the aggregations and that every chart can be drawn."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from michelin.analysis import stats
from michelin.config import AWARD_ORDER
from michelin.data.cleaning import clean_dataset
from michelin.viz import charts, maps


@pytest.fixture
def clean_rows(raw_rows):
    df, _ = clean_dataset(raw_rows)
    return df


def test_filter_dataset_combines_filters(clean_rows):
    result = stats.filter_dataset(clean_rows, countries=["France", "Italy"], price_levels=[4])
    assert result["Name"].tolist() == ["Alpha"]


def test_filter_dataset_without_filters_returns_everything(clean_rows):
    assert len(stats.filter_dataset(clean_rows)) == len(clean_rows)


def test_kpis(clean_rows):
    kpis = stats.compute_kpis(clean_rows)
    assert kpis.restaurants == 4
    assert kpis.starred == 2
    assert kpis.starred_share == pytest.approx(0.5)
    assert kpis.green_stars == 1


def test_award_distribution_keeps_order_and_zeros(clean_rows):
    distribution = stats.award_distribution(clean_rows)
    assert tuple(distribution.index) == AWARD_ORDER
    assert distribution["2 Stars"] == 0
    assert distribution.sum() == len(clean_rows)


def test_price_award_table_rows_sum_to_one(full_dataset):
    table = stats.price_award_table(full_dataset)
    assert np.allclose(table.sum(axis=1), 1.0)


def test_facility_comparison_shares_are_fractions(full_dataset):
    shares = stats.facility_comparison(full_dataset)
    assert list(shares.columns) == ["Starred", "Not starred"]
    assert ((shares >= 0) & (shares <= 1)).all().all()


def test_every_chart_renders(full_dataset):
    df = full_dataset.sample(1500, random_state=0)
    distribution = stats.award_distribution(df)
    figures = [
        charts.award_bar(distribution),
        charts.award_pie(distribution),
        charts.country_award_stacked(stats.country_award_table(df)),
        charts.ranking_bar(stats.top_cuisines(df), "t", "x"),
        charts.price_award_heatmap(stats.price_award_table(df)),
        charts.facility_comparison_bar(stats.facility_comparison(df)),
        charts.facility_count_box(df, "Award", "FacilityCount"),
        charts.correlation_heatmap(stats.numeric_correlation(df)),
        charts.confusion_matrix_heatmap(np.array([[5, 1], [2, 4]]), ["a", "b"], normalize=True),
    ]
    assert all(fig.axes for fig in figures)
    for fig in figures:
        plt.close(fig)

    assert maps.restaurant_map(df).data
    assert maps.starred_density_map(df).data


def test_labels_translate_awards_features_and_numbers():
    from michelin.labels import award_label, facility_label, feature_label, format_decimal, format_int

    assert award_label("Selected Restaurants") == "Selecionado"
    assert award_label("Starred") == "Estrelado"
    assert award_label("Unknown value") == "Unknown value"
    assert facility_label("Valet parking") == "Manobrista"
    assert feature_label("Has: Terrace") == "Tem: Terraço"
    assert feature_label("Country_France") == "País: França"
    assert feature_label("Cuisine_infrequent_sklearn") == "Culinária: outros (pouco frequentes)"
    assert feature_label("PriceLevel") == "Nível de preço"
    assert format_int(19985) == "19.985"
    assert format_decimal(0.7251) == "0,725"
