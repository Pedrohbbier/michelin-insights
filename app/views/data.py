"""Data page: upload files, see how they were cleaned and inspect the table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import state
from app.components import sidebar_status
from michelin.analysis.stats import data_quality_table
from michelin.config import (
    COL_AWARD,
    COL_COUNTRY,
    COL_FACILITY_LIST,
    REQUIRED_COLUMNS,
    SAMPLES_DIR,
    SUPPORTED_EXTENSIONS,
)
from michelin.data.cleaning import CleaningReport
from michelin.data.loader import UnsupportedFormatError, read_table
from michelin.data.schema import SchemaError, validate_schema
from michelin.labels import award_label, column_label, country_label, format_int

REPLACE = "Substituir os dados atuais"
APPEND = "Adicionar aos dados atuais"

DESCRIBE_LABELS = {
    "count": "contagem",
    "mean": "média",
    "std": "desvio padrão",
    "min": "mínimo",
    "25%": "25%",
    "50%": "mediana (50%)",
    "75%": "75%",
    "max": "máximo",
}


def _read_uploads(files: list) -> pd.DataFrame:
    """Read and validate every uploaded file, then stack them into one table."""
    frames = []
    for file in files:
        frame = read_table(file, file.name)
        validate_schema(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _upload_section() -> None:
    st.subheader("1. Carregar dados")
    st.write(
        "Envie um ou mais arquivos **CSV, JSON ou XML** com as colunas do Guia Michelin. "
        "Arquivos de outros países ou de edições mais novas do guia são analisados da mesma forma."
    )
    with st.expander("Colunas obrigatórias"):
        st.code(", ".join(REQUIRED_COLUMNS))

    files = st.file_uploader(
        "Arquivos de dados",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    mode = st.radio("Ao carregar", (REPLACE, APPEND), horizontal=True)

    left, right = st.columns(2)
    if left.button("Carregar arquivos enviados", type="primary", disabled=not files):
        try:
            uploaded = _read_uploads(files)
            names = ", ".join(file.name for file in files)
            if mode == APPEND:
                uploaded = pd.concat([state.get_raw_dataset(), uploaded], ignore_index=True)
                names = f"{state.get_dataset_info().source} + {names}"
            report = state.set_dataset(uploaded, names)
            st.session_state["last_load_message"] = f"{format_int(report.rows_out)} restaurantes carregados."
            st.rerun()
        except (SchemaError, UnsupportedFormatError, ValueError) as error:
            st.error(f"Não foi possível carregar os arquivos: {error}")

    if right.button("Voltar para a base completa"):
        state.load_default_dataset()
        st.session_state["last_load_message"] = "Base completa restaurada."
        st.rerun()

    message = st.session_state.pop("last_load_message", None)
    if message:
        st.success(message)
        if state.model_is_stale():
            st.info("Os dados mudaram. Vá em **Treinar** e retreine o modelo para atualizar as previsões.")

    _sample_downloads()


def _sample_downloads() -> None:
    samples = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix in SUPPORTED_EXTENSIONS)
    if not samples:
        return
    with st.expander("Arquivos de exemplo para testar o upload (um país por arquivo)"):
        columns = st.columns(3)
        for index, path in enumerate(samples):
            columns[index % 3].download_button(
                path.name,
                data=path.read_bytes(),
                file_name=path.name,
                key=f"sample_{path.name}",
                icon=":material/download:",
            )


def _report_section(report: CleaningReport) -> None:
    st.subheader("2. Relatório de limpeza")
    cols = st.columns(3)
    cols[0].metric("Linhas recebidas", format_int(report.rows_in))
    cols[1].metric("Linhas mantidas", format_int(report.rows_out))
    cols[2].metric("Linhas removidas", format_int(report.rows_removed))

    left, right = st.columns(2)
    with left:
        st.markdown("**Linhas removidas por motivo**")
        if report.removed:
            st.dataframe(pd.Series(report.removed, name="linhas").rename_axis("motivo"))
        else:
            st.caption("Nenhuma linha removida.")
    with right:
        st.markdown("**Valores vazios antes da limpeza**")
        if report.missing_before:
            missing = pd.Series(report.missing_before, name="vazios").rename(index=column_label)
            st.dataframe(missing.rename_axis("coluna"))
        else:
            st.caption("Nenhum valor vazio.")
    st.caption(
        "Etapas: remover espaços extras, excluir restaurantes duplicados, padronizar os nomes dos prêmios, "
        "converter os símbolos de preço em um nível de 1 a 4, validar as coordenadas, separar a localização "
        "em cidade e país e transformar o texto de serviços em uma lista."
    )


def _inspect_section(df: pd.DataFrame) -> None:
    st.subheader("3. Conferir os dados limpos")
    tab_preview, tab_info, tab_describe = st.tabs(["Prévia", "Colunas (info)", "Estatísticas (describe)"])
    with tab_preview:
        st.caption("As 200 primeiras linhas da tabela já limpa.")
        preview = df.drop(columns=[COL_FACILITY_LIST]).head(200)
        preview = preview.assign(
            **{COL_AWARD: preview[COL_AWARD].map(award_label), COL_COUNTRY: preview[COL_COUNTRY].map(country_label)}
        )
        st.dataframe(preview.rename(columns=column_label), hide_index=True)
    with tab_info:
        st.caption("Tipo de cada coluna, quantos valores estão preenchidos ou vazios e quantos valores diferentes há.")
        st.dataframe(data_quality_table(df).rename(index=column_label))
    with tab_describe:
        st.caption("Resumo estatístico das colunas numéricas, como o `df.describe()` do pandas.")
        summary = df.describe().T.rename(index=column_label, columns=DESCRIBE_LABELS)
        st.dataframe(summary)


def render() -> None:
    st.title("Dados")
    sidebar_status()
    _upload_section()
    st.divider()
    _report_section(state.get_cleaning_report())
    st.divider()
    _inspect_section(state.get_dataset())


render()
