"""Reusable Streamlit widgets shared by several pages."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from matplotlib.figure import Figure

from app import state
from app.descriptions import CHART_DESCRIPTIONS
from michelin.labels import choice_label, format_decimal, format_int
from michelin.ml.models import ModelSpec, ParamSpec


def _describe(description_key: str | None) -> None:
    if description_key:
        st.caption(CHART_DESCRIPTIONS[description_key])


def show_figure(fig: Figure, description_key: str | None = None) -> None:
    """Render a matplotlib figure with its description, then free its memory."""
    st.pyplot(fig)
    plt.close(fig)
    _describe(description_key)


def show_plotly(fig: go.Figure, description_key: str | None = None) -> None:
    """Render an interactive Plotly figure with its description."""
    st.plotly_chart(fig)
    _describe(description_key)


def sidebar_status() -> None:
    """Dataset and model summary shown on every page."""
    df = state.get_dataset()
    info = state.get_dataset_info()
    model = state.get_model()
    with st.sidebar:
        st.markdown("#### Dados ativos")
        st.caption(f"{info.source}  \n{format_int(len(df))} restaurantes · carregados às {info.loaded_at:%H:%M:%S}")
        st.markdown("#### Modelo")
        if model is None:
            st.caption("Nenhum modelo treinado ainda.")
        else:
            st.caption(
                f"{model.config.model_spec.label}  \nAcurácia no teste: {format_decimal(model.accuracy)}"
            )
            if state.model_is_stale():
                st.warning("Os dados mudaram depois do treino. Retreine o modelo na página **Treinar**.")


def param_widget(param: ParamSpec, key_prefix: str) -> Any:
    """Draw the right input for a hyperparameter and return the value."""
    key = f"{key_prefix}_{param.name}"
    if param.kind == "choice":
        options = list(param.options)
        value = st.selectbox(
            param.label, options, index=options.index(param.default), format_func=choice_label, key=key
        )
    elif param.kind == "int":
        value = st.slider(
            param.label,
            min_value=int(param.min_value),
            max_value=int(param.max_value),
            value=int(param.default),
            step=int(param.step or 1),
            key=key,
        )
    else:
        value = st.number_input(
            param.label,
            min_value=float(param.min_value),
            max_value=float(param.max_value),
            value=float(param.default),
            step=float(param.step or 0.1),
            key=key,
        )
    if param.help:
        st.caption(param.help)
    return value


def model_params_form(spec: ModelSpec) -> dict[str, Any]:
    return {param.name: param_widget(param, spec.key) for param in spec.params}


def price_label(level: int) -> str:
    return "$" * int(level)
