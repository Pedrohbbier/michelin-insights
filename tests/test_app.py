"""Smoke tests that run every page of the Streamlit app headlessly."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
TIMEOUT = 120


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    return at


@pytest.mark.parametrize(
    "page", ["app/views/data.py", "app/views/explore.py", "app/views/train.py", "app/views/predict.py"]
)
def test_page_renders_without_errors(app, page):
    app.switch_page(page).run()
    assert not app.exception


def test_train_then_predict(app):
    app.switch_page("app/views/train.py").run()
    # A single decision tree keeps the test fast.
    app.selectbox[0].set_value("decision_tree").run()
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == "Acurácia" for metric in app.metric)

    app.switch_page("app/views/predict.py").run()
    app.button[0].click().run()  # Submit the form with default values.
    assert not app.exception
    assert any(metric.label == "Prêmio mais provável" for metric in app.metric)
