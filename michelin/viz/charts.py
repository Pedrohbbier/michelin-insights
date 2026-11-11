"""Static charts built with matplotlib and seaborn.

Every function returns a ``matplotlib.figure.Figure`` so the caller decides
how to show it (``st.pyplot`` in the app, ``plt.show`` in a notebook).
Data values stay in English; labels are translated with :mod:`michelin.labels`.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # Render off-screen; the web app never opens a window.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from michelin.config import AWARD_COLORS
from michelin.labels import (
    award_label,
    award_labels,
    country_label,
    facility_label,
    format_int,
    format_percent,
)

sns.set_theme(style="whitegrid", context="notebook")

_NEUTRAL = "#4C6A92"
_PERCENT_AXIS = FuncFormatter(lambda value, _: format_percent(value))
_INT_AXIS = FuncFormatter(lambda value, _: format_int(value))


def _colors_for(labels: Sequence[str]) -> list[str]:
    return [AWARD_COLORS.get(str(label), _NEUTRAL) for label in labels]


def award_bar(distribution: pd.Series) -> Figure:
    """Bar chart with the number of restaurants per award."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(award_labels(distribution.index), distribution.values, color=_colors_for(distribution.index))
    ax.bar_label(bars, labels=[format_int(v) for v in distribution.values], padding=3)
    ax.set_title("Restaurantes por prêmio")
    ax.set_ylabel("Restaurantes")
    ax.yaxis.set_major_formatter(_INT_AXIS)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def award_pie(distribution: pd.Series) -> Figure:
    """Pie chart with the share of each award (5 slices at most)."""
    data = distribution[distribution > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        data.values,
        labels=award_labels(data.index),
        colors=_colors_for(data.index),
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Participação de cada prêmio")
    fig.tight_layout()
    return fig


def country_award_stacked(table: pd.DataFrame) -> Figure:
    """Horizontal stacked bars: award mix of the countries with most restaurants."""
    ordered = table.iloc[::-1]  # Largest country on top.
    colors = _colors_for(ordered.columns)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(ordered))))
    ordered.rename(index=country_label, columns=award_label).plot(
        kind="barh", stacked=True, ax=ax, color=colors, width=0.8
    )
    ax.set_title("Prêmios por país")
    ax.set_xlabel("Restaurantes")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(_INT_AXIS)
    ax.legend(title="Prêmio", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def ranking_bar(series: pd.Series, title: str, xlabel: str, color: str = AWARD_COLORS["1 Star"]) -> Figure:
    """Generic horizontal ranking chart (top countries, cities, cuisines...)."""
    ordered = series.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(ordered))))
    bars = ax.barh(ordered.index.astype(str), ordered.values, color=color)
    ax.bar_label(bars, labels=[format_int(v) for v in ordered.values], padding=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(_INT_AXIS)
    fig.tight_layout()
    return fig


def price_award_heatmap(table: pd.DataFrame) -> Figure:
    """Heatmap of the award share at each price level."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    # "$" starts a math formula in matplotlib, so it must be escaped.
    labels = [r"\$" * int(level) for level in table.index]
    sns.heatmap(
        table.rename(columns=award_label),
        annot=True,
        fmt=".0%",
        cmap="Reds",
        cbar=False,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Distribuição dos prêmios em cada faixa de preço")
    ax.set_xlabel("")
    ax.set_ylabel("Preço")
    fig.tight_layout()
    return fig


def facility_comparison_bar(shares: pd.DataFrame) -> Figure:
    """Grouped bars: how often starred vs non-starred places offer each facility."""
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(shares))))
    display = shares.rename(index=facility_label, columns=award_label)
    display.plot(
        kind="barh",
        ax=ax,
        color=[AWARD_COLORS["Starred"], AWARD_COLORS["Not starred"]],
        width=0.8,
    )
    ax.xaxis.set_major_formatter(_PERCENT_AXIS)
    ax.set_title("Serviços: estrelados x não estrelados")
    ax.set_xlabel("% dos restaurantes que oferecem")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def facility_count_box(df: pd.DataFrame, award_col: str, count_col: str) -> Figure:
    """Boxplot of the number of listed facilities for each award."""
    fig, ax = plt.subplots(figsize=(7, 4))
    awards = [a for a in df[award_col].cat.categories if (df[award_col] == a).any()]
    sns.boxplot(
        data=df,
        x=award_col,
        y=count_col,
        order=awards,
        hue=award_col,
        hue_order=awards,
        palette=_colors_for(awards),
        legend=False,
        ax=ax,
    )
    ax.set_xticks(range(len(awards)), award_labels(awards))
    ax.set_title("Quantidade de serviços por prêmio")
    ax.set_xlabel("")
    ax.set_ylabel("Serviços listados")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlação entre as variáveis numéricas")
    fig.tight_layout()
    return fig


# --- Machine learning charts -------------------------------------------------


def confusion_matrix_heatmap(matrix: np.ndarray, labels: Sequence[str], normalize: bool = False) -> Figure:
    """Confusion matrix: rows are true classes, columns are predictions."""
    values = matrix.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    names = award_labels(labels)
    size = max(4.0, 1.1 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(
        values,
        annot=True,
        fmt=".0%" if normalize else ".0f",
        cmap="Blues",
        cbar=False,
        xticklabels=names,
        yticklabels=names,
        ax=ax,
    )
    ax.set_xlabel("Previsto pelo modelo")
    ax.set_ylabel("Prêmio real")
    ax.set_title("Matriz de confusão" + (" (% por linha)" if normalize else ""))
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def probability_bar(probabilities: pd.Series) -> Figure:
    """Predicted probability for each class of a single restaurant."""
    ordered = probabilities.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.5 * len(ordered))))
    bars = ax.barh(award_labels(ordered.index), ordered.values, color=_colors_for(ordered.index))
    ax.bar_label(bars, labels=[format_percent(p) for p in ordered.values], padding=3)
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(_PERCENT_AXIS)
    ax.set_title("Probabilidade de cada prêmio")
    fig.tight_layout()
    return fig
