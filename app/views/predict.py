"""Predict page: describe a restaurant and get the most likely award."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import state
from app.components import price_label, show_figure, sidebar_status
from app.descriptions import CHART_DESCRIPTIONS
from michelin.config import COL_AWARD, COL_CITY, COL_COUNTRY, COL_CUISINE, COL_NAME, COL_PRICE, COL_PRICE_LEVEL
from michelin.labels import award_label, column_label, country_label, facility_label, format_decimal, format_percent
from michelin.ml.predict import RestaurantInput, predict_restaurant
from michelin.ml.training import TrainingResult
from michelin.viz import charts

PRICE_LEVELS = (1, 2, 3, 4)


def _default_index(options: tuple[str, ...], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def _input_form(result: TrainingResult) -> RestaurantInput | None:
    with st.form("restaurant"):
        left, right = st.columns(2)
        with left:
            countries = tuple(sorted(result.countries, key=country_label))
            country = st.selectbox(
                "País", countries, index=_default_index(countries, "France"), format_func=country_label
            )
            cuisine = st.selectbox(
                "Culinária", result.cuisines, index=_default_index(result.cuisines, "Modern Cuisine")
            )
            price_level = st.select_slider("Preço", PRICE_LEVELS, value=3, format_func=price_label)
        with right:
            facilities = st.multiselect(
                "Serviços oferecidos",
                result.feature_spec.facilities,
                format_func=facility_label,
                placeholder="Escolha os serviços",
            )
            green_star = st.checkbox("Tem Estrela Verde (gastronomia sustentável)")
        submitted = st.form_submit_button("Prever", type="primary")
    if not submitted:
        return None
    return RestaurantInput(country, cuisine, price_level, facilities, green_star)


def _price_sensitivity(result: TrainingResult, restaurant: RestaurantInput) -> pd.DataFrame:
    """Probabilities for the same restaurant at every price level."""
    rows = {}
    for level in PRICE_LEVELS:
        variant = RestaurantInput(
            restaurant.country, restaurant.cuisine, level, restaurant.facilities, restaurant.green_star
        )
        rows[price_label(level)] = predict_restaurant(result, variant).probabilities
    return pd.DataFrame(rows).T.rename(columns=award_label)


def _similar_restaurants(restaurant: RestaurantInput) -> pd.DataFrame:
    df = state.get_dataset()
    similar = df[
        (df[COL_COUNTRY] == restaurant.country)
        & (df[COL_CUISINE] == restaurant.cuisine)
        & (df[COL_PRICE_LEVEL] == restaurant.price_level)
    ]
    table = similar[[COL_NAME, COL_CITY, COL_PRICE, COL_AWARD]].head(20)
    table = table.assign(**{COL_AWARD: table[COL_AWARD].map(award_label)})
    return table.rename(columns=column_label)


def render() -> None:
    st.title("Prever")
    sidebar_status()
    result = state.get_model()
    if result is None:
        st.info("Ainda não há modelo. Treine um na página **Treinar** primeiro.")
        st.page_link("app/views/train.py", label="Ir para Treinar", icon=":material/model_training:")
        return

    st.write(
        f"Modelo: **{result.config.model_spec.label}** "
        f"(acurácia no teste: {format_decimal(result.accuracy)})."
    )
    st.caption("Descreva um restaurante e o modelo estima qual prêmio ele provavelmente receberia.")
    if state.model_is_stale():
        st.warning("Os dados mudaram depois do treino deste modelo. Retreine para usar os dados novos.")

    restaurant = _input_form(result)
    if restaurant is None:
        return

    prediction = predict_restaurant(result, restaurant)
    left, right = st.columns([2, 3])
    with left:
        st.metric(
            "Prêmio mais provável",
            award_label(prediction.label),
            f"{format_percent(prediction.probabilities.max())} de probabilidade",
            delta_color="off",
        )
        show_figure(charts.probability_bar(prediction.probabilities), "probability")
    with right:
        st.markdown("**Como o preço muda a previsão**")
        sensitivity = _price_sensitivity(result, restaurant)
        st.bar_chart(sensitivity, stack=True, y_label="Probabilidade", x_label="Preço")
        st.caption(CHART_DESCRIPTIONS["price_sensitivity"])

    st.markdown("**Restaurantes reais com o mesmo país, culinária e preço**")
    similar = _similar_restaurants(restaurant)
    if similar.empty:
        st.caption("Nenhum nos dados ativos.")
    else:
        st.caption("Servem para comparar a previsão com o que aconteceu de verdade no guia.")
        st.dataframe(similar, hide_index=True)


render()
