"""Explore page: filters, KPIs, charts and maps for the active dataset."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import state
from app.components import price_label, show_figure, show_plotly, sidebar_status
from michelin.analysis import stats
from michelin.config import (
    AWARD_ORDER,
    COL_AWARD,
    COL_CITY,
    COL_COUNTRY,
    COL_FACILITY_COUNT,
    COL_FACILITY_LIST,
    COL_PRICE_LEVEL,
)
from michelin.labels import award_label, column_label, country_label, format_int, format_percent
from michelin.viz import charts, maps

# Plotting thousands of points is slow in the browser; sample above this size.
MAX_MAP_POINTS = 8000
MAP_BY_AWARD = "Restaurantes por prêmio"
MAP_DENSITY = "Onde estão as estrelas"


def _filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown("#### Filtros")
        countries = st.multiselect(
            "País",
            sorted(df[COL_COUNTRY].unique(), key=country_label),
            format_func=country_label,
            placeholder="Todos os países",
        )
        awards = st.multiselect(
            "Prêmio",
            [a for a in AWARD_ORDER if (df[COL_AWARD] == a).any()],
            format_func=award_label,
            placeholder="Todos os prêmios",
        )
        levels = sorted(df[COL_PRICE_LEVEL].unique())
        prices = st.multiselect("Preço", levels, format_func=price_label, placeholder="Todos os preços")
    return stats.filter_dataset(df, countries, awards, prices)


def _kpis(df: pd.DataFrame) -> None:
    kpis = stats.compute_kpis(df)
    cols = st.columns(5)
    cols[0].metric("Restaurantes", format_int(kpis.restaurants))
    cols[1].metric("Países", kpis.countries)
    cols[2].metric("Cidades", format_int(kpis.cities))
    cols[3].metric(
        "Estrelados",
        format_int(kpis.starred),
        f"{format_percent(kpis.starred_share, 1)} do total",
        delta_color="off",
    )
    cols[4].metric("Estrelas Verdes", kpis.green_stars, help="Prêmio do guia para gastronomia sustentável.")


def _overview_tab(df: pd.DataFrame) -> None:
    distribution = stats.award_distribution(df)
    left, right = st.columns([3, 2])
    with left:
        show_figure(charts.award_bar(distribution), "award_bar")
    with right:
        show_figure(charts.award_pie(distribution), "award_pie")
    st.markdown("**Preço mais alto significa prêmio melhor?**")
    show_figure(charts.price_award_heatmap(stats.price_award_table(df)), "price_heatmap")


def _geography_tab(df: pd.DataFrame) -> None:
    map_data = df.sample(MAX_MAP_POINTS, random_state=0) if len(df) > MAX_MAP_POINTS else df
    view = st.segmented_control("Mapa", [MAP_BY_AWARD, MAP_DENSITY], default=MAP_BY_AWARD)
    if view == MAP_DENSITY:
        show_plotly(maps.starred_density_map(df), "density_map")
    else:
        if len(map_data) < len(df):
            st.caption(
                f"Mostrando uma amostra aleatória de {format_int(MAX_MAP_POINTS)} restaurantes. "
                "Filtre por país para ver todos."
            )
        show_plotly(maps.restaurant_map(map_data), "restaurant_map")

    left, right = st.columns(2)
    with left:
        show_figure(charts.country_award_stacked(stats.country_award_table(df, top_n=15)), "country_stacked")
    with right:
        ranking = stats.starred_ranking(df, by=COL_CITY, top_n=15)
        if ranking.empty:
            st.info("Nenhum restaurante estrelado no filtro atual.")
        else:
            figure = charts.ranking_bar(ranking, "Cidades com mais restaurantes estrelados", "Restaurantes estrelados")
            show_figure(figure, "city_ranking")


def _cuisine_tab(df: pd.DataFrame) -> None:
    starred_only = st.toggle("Somente restaurantes estrelados", value=False)
    cuisines = stats.top_cuisines(df, top_n=15, starred_only=starred_only)
    if cuisines.empty:
        st.info("Nenhum restaurante corresponde ao filtro atual.")
    else:
        figure = charts.ranking_bar(cuisines, "Culinárias mais comuns", "Restaurantes", color="#4C6A92")
        show_figure(figure, "cuisine_ranking")

    left, right = st.columns(2)
    with left:
        comparison = stats.facility_comparison(df)
        if not comparison.empty:
            show_figure(charts.facility_comparison_bar(comparison), "facility_comparison")
    with right:
        show_figure(charts.facility_count_box(df, COL_AWARD, COL_FACILITY_COUNT), "facility_box")


def _correlation_tab(df: pd.DataFrame) -> None:
    left, _ = st.columns([3, 2])
    with left:
        show_figure(charts.correlation_heatmap(stats.numeric_correlation(df)), "correlation")


def _table_tab(df: pd.DataFrame) -> None:
    table = df.drop(columns=[COL_FACILITY_LIST])
    st.caption("Todas as linhas que passam pelos filtros da barra lateral.")
    display = table.assign(
        **{COL_AWARD: table[COL_AWARD].map(award_label), COL_COUNTRY: table[COL_COUNTRY].map(country_label)}
    )
    st.dataframe(display.rename(columns=column_label), hide_index=True)
    st.download_button(
        "Baixar dados filtrados (CSV)",
        table.to_csv(index=False).encode(),
        "michelin_filtrado.csv",
        "text/csv",
        help="O arquivo mantém os nomes de coluna originais, então pode ser enviado de volta na página Dados.",
    )


def render() -> None:
    st.title("Explorar")
    sidebar_status()
    df = _filters(state.get_dataset())
    if df.empty:
        st.warning("Nenhum restaurante corresponde aos filtros.")
        return

    _kpis(df)
    tabs = st.tabs(["Visão geral", "Geografia", "Culinária e serviços", "Correlação", "Tabela"])
    with tabs[0]:
        _overview_tab(df)
    with tabs[1]:
        _geography_tab(df)
    with tabs[2]:
        _cuisine_tab(df)
    with tabs[3]:
        _correlation_tab(df)
    with tabs[4]:
        _table_tab(df)


render()
