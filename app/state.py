"""Typed access to Streamlit's session state.

Pages never touch ``st.session_state`` keys directly; they go through these
helpers so the keys and their types live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st

from michelin.config import DEFAULT_DATASET
from michelin.data.cleaning import CleaningReport, clean_dataset
from michelin.data.loader import read_path
from michelin.ml.training import TrainingResult, dataset_fingerprint

_RAW = "raw_dataset"
_CLEAN = "dataset"
_SOURCE = "dataset_source"
_REPORT = "cleaning_report"
_MODEL = "model"

DEFAULT_SOURCE = "Base completa do Guia Michelin (incluída no projeto)"


@dataclass(frozen=True)
class DatasetInfo:
    source: str
    loaded_at: datetime


@st.cache_data(show_spinner="Carregando a base do Guia Michelin...")
def _load_default_raw() -> pd.DataFrame:
    return read_path(DEFAULT_DATASET)


def set_dataset(raw: pd.DataFrame, source: str) -> CleaningReport:
    """Clean ``raw`` and make it the active dataset. Raises SchemaError on bad input."""
    clean, report = clean_dataset(raw)
    st.session_state[_RAW] = raw
    st.session_state[_CLEAN] = clean
    st.session_state[_REPORT] = report
    st.session_state[_SOURCE] = DatasetInfo(source=source, loaded_at=datetime.now())
    return report


def ensure_dataset() -> None:
    """Load the bundled dataset the first time the app opens."""
    if _CLEAN not in st.session_state:
        set_dataset(_load_default_raw(), DEFAULT_SOURCE)


def load_default_dataset() -> CleaningReport:
    return set_dataset(_load_default_raw(), DEFAULT_SOURCE)


def get_raw_dataset() -> pd.DataFrame:
    ensure_dataset()
    return st.session_state[_RAW]


def get_dataset() -> pd.DataFrame:
    ensure_dataset()
    return st.session_state[_CLEAN]


def get_dataset_info() -> DatasetInfo:
    ensure_dataset()
    return st.session_state[_SOURCE]


def get_cleaning_report() -> CleaningReport:
    ensure_dataset()
    return st.session_state[_REPORT]


def get_model() -> TrainingResult | None:
    return st.session_state.get(_MODEL)


def set_model(result: TrainingResult) -> None:
    st.session_state[_MODEL] = result


def model_is_stale() -> bool:
    """True when the active dataset changed after the current model was trained."""
    model = get_model()
    return model is not None and model.data_fingerprint != dataset_fingerprint(get_dataset())
