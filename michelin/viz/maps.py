"""Interactive maps built with Plotly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from michelin.config import (
    AWARD_COLORS,
    AWARD_ORDER,
    COL_AWARD,
    COL_CITY,
    COL_COUNTRY,
    COL_CUISINE,
    COL_IS_STARRED,
    COL_LATITUDE,
    COL_LONGITUDE,
    COL_NAME,
    COL_PRICE,
)
from michelin.labels import COLUMN_LABELS, award_label, country_label

_MAP_STYLE = "carto-positron"
_MAP_HEIGHT = 600


def _view_for(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Center and zoom so every point fits on screen."""
    lat, lon = df[COL_LATITUDE], df[COL_LONGITUDE]
    center = {"lat": float(lat.mean()), "lon": float(lon.mean())}
    span = max(float(lat.max() - lat.min()), float(lon.max() - lon.min()), 0.01)
    # Roughly: zoom 0 shows 360 degrees, each zoom level halves the span.
    zoom = float(np.clip(np.log2(360 / span) - 0.5, 0.5, 12))
    return center, zoom


def restaurant_map(df: pd.DataFrame) -> go.Figure:
    """Scatter map with one point per restaurant, colored by award."""
    center, zoom = _view_for(df)
    present = [award for award in AWARD_ORDER if (df[COL_AWARD] == award).any()]
    colors = {award_label(award): color for award, color in AWARD_COLORS.items()}
    fig = px.scatter_map(
        df.assign(
            **{
                COL_AWARD: df[COL_AWARD].map(award_label).astype(str),
                COL_COUNTRY: df[COL_COUNTRY].map(country_label),
            }
        ),
        lat=COL_LATITUDE,
        lon=COL_LONGITUDE,
        color=COL_AWARD,
        category_orders={COL_AWARD: [award_label(a) for a in present]},
        color_discrete_map=colors,
        hover_name=COL_NAME,
        hover_data={
            COL_CITY: True,
            COL_COUNTRY: True,
            COL_CUISINE: True,
            COL_PRICE: True,
            COL_LATITUDE: False,
            COL_LONGITUDE: False,
        },
        labels=COLUMN_LABELS,
        center=center,
        zoom=zoom,
        height=_MAP_HEIGHT,
        map_style=_MAP_STYLE,
        opacity=0.75,
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend={"title": "Prêmio"})
    return fig


def starred_density_map(df: pd.DataFrame) -> go.Figure:
    """Heat map showing where starred restaurants concentrate."""
    starred = df[df[COL_IS_STARRED]]
    data = starred if not starred.empty else df
    center, zoom = _view_for(data)
    fig = px.density_map(
        data,
        lat=COL_LATITUDE,
        lon=COL_LONGITUDE,
        radius=12,
        hover_name=COL_NAME,
        labels=COLUMN_LABELS,
        center=center,
        zoom=zoom,
        height=_MAP_HEIGHT,
        map_style=_MAP_STYLE,
        color_continuous_scale="Reds",
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, coloraxis_showscale=False)
    return fig
