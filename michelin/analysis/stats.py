"""Pandas aggregations behind the charts.

These functions only compute tables; drawing happens in :mod:`michelin.viz`.
Keeping them apart makes the numbers easy to test and reuse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from michelin.config import (
    AWARD_ORDER,
    COL_AWARD,
    COL_CITY,
    COL_COUNTRY,
    COL_CUISINE,
    COL_FACILITY_COUNT,
    COL_FACILITY_LIST,
    COL_GREEN_STAR,
    COL_IS_STARRED,
    COL_LATITUDE,
    COL_PRICE_LEVEL,
)


@dataclass(frozen=True)
class Kpis:
    restaurants: int
    countries: int
    cities: int
    starred: int
    starred_share: float
    green_stars: int
    mean_price_level: float


def filter_dataset(
    df: pd.DataFrame,
    countries: Sequence[str] | None = None,
    awards: Sequence[str] | None = None,
    price_levels: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Return the rows matching every non-empty filter."""
    mask = pd.Series(True, index=df.index)
    if countries:
        mask &= df[COL_COUNTRY].isin(countries)
    if awards:
        mask &= df[COL_AWARD].isin(awards)
    if price_levels:
        mask &= df[COL_PRICE_LEVEL].isin(price_levels)
    return df[mask]


def compute_kpis(df: pd.DataFrame) -> Kpis:
    starred = int(df[COL_IS_STARRED].sum())
    total = len(df)
    return Kpis(
        restaurants=total,
        countries=df[COL_COUNTRY].nunique(),
        cities=df[COL_CITY].nunique(),
        starred=starred,
        starred_share=starred / total if total else 0.0,
        green_stars=int(df[COL_GREEN_STAR].sum()),
        mean_price_level=float(df[COL_PRICE_LEVEL].mean()) if total else 0.0,
    )


def award_distribution(df: pd.DataFrame) -> pd.Series:
    """Number of restaurants per award, in award order (including zeros)."""
    return df[COL_AWARD].value_counts().reindex(AWARD_ORDER, fill_value=0)


def country_award_table(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Awards per country for the ``top_n`` countries with the most restaurants."""
    table = pd.crosstab(df[COL_COUNTRY], df[COL_AWARD]).reindex(columns=AWARD_ORDER, fill_value=0)
    totals = table.sum(axis=1).sort_values(ascending=False)
    return table.loc[totals.index[:top_n]]


def starred_ranking(df: pd.DataFrame, by: str = COL_COUNTRY, top_n: int = 15) -> pd.Series:
    """Count of starred restaurants grouped by ``by`` (country or city)."""
    starred = df[df[COL_IS_STARRED]]
    return starred[by].value_counts().head(top_n)


def price_award_table(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Crosstab of price level (rows) by award (columns).

    With ``normalize=True`` each row sums to 1, showing which awards are
    common at each price level.
    """
    table = pd.crosstab(
        df[COL_PRICE_LEVEL],
        df[COL_AWARD],
        normalize="index" if normalize else False,
    )
    return table.reindex(columns=AWARD_ORDER, fill_value=0)


def top_cuisines(df: pd.DataFrame, top_n: int = 15, starred_only: bool = False) -> pd.Series:
    data = df[df[COL_IS_STARRED]] if starred_only else df
    return data[COL_CUISINE].value_counts().head(top_n)


def facility_comparison(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Share of starred vs non-starred restaurants offering each facility.

    Returns a DataFrame indexed by facility with columns "Starred" and
    "Not starred", sorted by how much more common the facility is among
    starred restaurants.
    """
    exploded = df[[COL_FACILITY_LIST, COL_IS_STARRED]].explode(COL_FACILITY_LIST).dropna()
    group_sizes = df[COL_IS_STARRED].value_counts()
    counts = exploded.groupby([COL_FACILITY_LIST, COL_IS_STARRED]).size().unstack(fill_value=0)
    shares = counts.div(group_sizes, axis="columns").fillna(0)
    shares = shares.rename(columns={True: "Starred", False: "Not starred"})
    for column in ("Starred", "Not starred"):
        if column not in shares:
            shares[column] = 0.0
    most_common = shares.max(axis=1).sort_values(ascending=False).index[:top_n]
    shares = shares.loc[most_common, ["Starred", "Not starred"]]
    return shares.assign(gap=shares["Starred"] - shares["Not starred"]).sort_values("gap").drop(columns="gap")


def numeric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between the numeric features and the award rank."""
    numeric = pd.DataFrame(
        {
            "Nível de preço": df[COL_PRICE_LEVEL],
            "Qtd. de serviços": df[COL_FACILITY_COUNT],
            "Estrela Verde": df[COL_GREEN_STAR],
            "Latitude": df[COL_LATITUDE],
            "Nível do prêmio": df[COL_AWARD].cat.codes,
        }
    )
    return numeric.corr()


def data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column overview similar to ``df.info()``: type, nulls and unique values."""
    visible = df.drop(columns=[COL_FACILITY_LIST], errors="ignore")
    return pd.DataFrame(
        {
            "tipo": visible.dtypes.astype(str),
            "preenchidos": visible.notna().sum(),
            "vazios": visible.isna().sum(),
            "valores únicos": visible.nunique(),
        }
    )
