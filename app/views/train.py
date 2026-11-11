"""Train page: pick a classifier, tune it, train or retrain, and read the evaluation."""

from __future__ import annotations

import streamlit as st

from app import state
from app.components import model_params_form, show_figure, sidebar_status
from michelin.labels import format_decimal, format_int
from michelin.ml.models import MODEL_REGISTRY, get_model_spec
from michelin.ml.training import TrainingConfig, train_model
from michelin.viz import charts


def _options_summary() -> None:
    """Explain what each remaining option changes in the prediction."""
    st.markdown(
        "O modelo calcula a **probabilidade de cada prêmio** para um restaurante, e a maior vira a previsão "
        "(ex.: 20% Selecionado, 6% Bib Gourmand, **74% Estrelado**). Cada opção abaixo muda uma parte disso:"
    )
    st.markdown(
        "| Opção | O que muda |\n"
        "|---|---|\n"
        "| **Classificador** | O jeito de calcular a probabilidade. |\n"
        "| **Parâmetros** (profundidade, K...) | Ajustam esse jeito. Ex.: K=5 consulta 5 vizinhos e K=50 consulta "
        "50, o que gera porcentagens diferentes. |"
    )


def _config_form() -> TrainingConfig:
    st.subheader("1. Configurar")
    _options_summary()
    model_key = st.selectbox(
        "Classificador",
        list(MODEL_REGISTRY),
        format_func=lambda key: MODEL_REGISTRY[key].label,
    )
    spec = get_model_spec(model_key)
    st.caption(spec.description)
    params = model_params_form(spec)
    return TrainingConfig(model_key=model_key, params=params)


def _train_button(config: TrainingConfig) -> None:
    df = state.get_dataset()
    has_model = state.get_model() is not None
    label = "Retreinar modelo" if has_model else "Treinar modelo"
    st.caption(f"O treino usa os dados ativos: **{format_int(len(df))} restaurantes**.")
    if state.model_is_stale():
        st.warning("Novos dados foram carregados depois do último treino. Retreine para atualizar o modelo.")
    if st.button(label, type="primary", icon=":material/model_training:"):
        try:
            with st.spinner("Treinando..."):
                state.set_model(train_model(df, config))
        except ValueError as error:
            st.error(str(error))


def _scores_section() -> None:
    result = state.get_model()
    if result is None:
        return
    st.subheader("2. Avaliação")
    st.metric(
        "Acurácia",
        format_decimal(result.accuracy),
        help="Porcentagem de acertos no conjunto de teste (20% dos dados).",
    )
    st.caption(
        f"O modelo estudou {format_int(result.n_train)} restaurantes (80%) e fez a prova com "
        f"{format_int(result.n_test)} que ele nunca viu (20%). A acurácia é a nota dessa prova."
    )
    normalize = st.toggle("Mostrar a matriz de confusão em % por linha", value=True)
    figure = charts.confusion_matrix_heatmap(result.confusion, result.labels, normalize=normalize)
    show_figure(figure, "confusion_matrix")


def render() -> None:
    st.title("Treinar")
    sidebar_status()
    config = _config_form()
    _train_button(config)

    result = state.get_model()
    if result is None:
        st.info("Treine um modelo para ver a avaliação.")
        return

    st.divider()
    st.caption(f"Exibindo: **{result.config.model_spec.label}** · treinado às {result.trained_at:%H:%M:%S}")
    _scores_section()


render()
