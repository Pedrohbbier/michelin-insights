"""Entry point: ``streamlit run streamlit_app.py``.

Sets up the page and the navigation; each page lives in ``app/views``.
"""

import streamlit as st

from app import state

st.set_page_config(page_title="Michelin Insights", page_icon=":material/restaurant:", layout="wide")

# Load the bundled Michelin CSV on first open so every page has data ready.
state.ensure_dataset()

# Sidebar order matches the usual flow: load data → explore → train → predict.
pages = [
    st.Page("app/views/data.py", title="Dados", icon=":material/upload_file:", default=True),
    st.Page("app/views/explore.py", title="Explorar", icon=":material/insights:"),
    st.Page("app/views/train.py", title="Treinar", icon=":material/model_training:"),
    st.Page("app/views/predict.py", title="Prever", icon=":material/online_prediction:"),
]

st.navigation(pages).run()
