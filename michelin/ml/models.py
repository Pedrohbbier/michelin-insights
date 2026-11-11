"""Catalog of the classifiers the user can pick, with their tunable parameters.

The UI reads :data:`MODEL_REGISTRY` to draw one widget per parameter, so
adding a new model only requires a new entry here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from michelin.config import RANDOM_STATE

ParamKind = Literal["int", "float", "choice"]


@dataclass(frozen=True)
class ParamSpec:
    """One hyperparameter and the range the user may choose from."""

    name: str
    label: str
    kind: ParamKind
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: Sequence[Any] = ()
    help: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """A classifier option shown in the app."""

    key: str
    label: str
    description: str
    build: Callable[[Mapping[str, Any]], ClassifierMixin]
    params: tuple[ParamSpec, ...] = field(default_factory=tuple)

    def defaults(self) -> dict[str, Any]:
        return {param.name: param.default for param in self.params}


def _knn(params: Mapping[str, Any]) -> ClassifierMixin:
    return KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]), weights=params["weights"])


def _decision_tree(params: Mapping[str, Any]) -> ClassifierMixin:
    return DecisionTreeClassifier(
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        criterion=params["criterion"],
        random_state=RANDOM_STATE,
    )


def _random_forest(params: Mapping[str, Any]) -> ClassifierMixin:
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _logistic_regression(params: Mapping[str, Any]) -> ClassifierMixin:
    return LogisticRegression(C=float(params["C"]), max_iter=2000, random_state=RANDOM_STATE)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="decision_tree",
            label="Árvore de Decisão",
            description=(
                'Responde uma sequência de perguntas (ex.: "o preço é \\$\\$\\$\\$?", "tem carta de vinhos?") '
                "até o fim de um caminho. A probabilidade é a porcentagem de cada prêmio entre os restaurantes "
                "de treino que chegaram ao mesmo ponto. Ex.: se 74% deles eram estrelados, a chance de estrela é 74%."
            ),
            build=_decision_tree,
            params=(
                ParamSpec(
                    "max_depth",
                    "Profundidade máxima",
                    "int",
                    8,
                    1,
                    30,
                    1,
                    help=(
                        "Quantas perguntas seguidas a árvore pode fazer. Ex.: com 2, ela só consegue perguntar "
                        "o preço e mais uma coisa; com 20, cria caminhos tão específicos que podem valer só para "
                        "poucos restaurantes do treino e errar com restaurantes novos."
                    ),
                ),
                ParamSpec(
                    "min_samples_leaf",
                    "Mínimo de restaurantes por folha",
                    "int",
                    5,
                    1,
                    50,
                    1,
                    help=(
                        "Quantos restaurantes, no mínimo, precisam chegar ao fim de cada caminho. Ex.: com 1, "
                        "um único restaurante pode virar regra (100% de chance baseado em 1 caso); com 50, "
                        "a porcentagem é calculada sobre pelo menos 50 restaurantes, o que é mais confiável."
                    ),
                ),
                ParamSpec(
                    "criterion",
                    "Critério de divisão",
                    "choice",
                    "gini",
                    options=("gini", "entropy"),
                    help=(
                        "A fórmula que escolhe qual pergunta fazer em cada passo. Gini e Entropia costumam "
                        "gerar árvores e porcentagens quase iguais; vale testar os dois e comparar o resultado."
                    ),
                ),
            ),
        ),
        ModelSpec(
            key="random_forest",
            label="Random Forest (Floresta Aleatória)",
            description=(
                "Cria várias árvores de decisão, cada uma estudando uma amostra sorteada dos restaurantes, "
                'e faz uma votação. Ex.: se 148 de 200 árvores dizem "Estrelado", a chance de estrela é 74%.'
            ),
            build=_random_forest,
            params=(
                ParamSpec(
                    "n_estimators",
                    "Número de árvores",
                    "int",
                    200,
                    10,
                    500,
                    10,
                    help=(
                        "Quantas árvores votam. Ex.: com 10, a porcentagem muda bastante a cada treino "
                        "(cada voto vale 10%); com 200, fica estável. Mais árvores deixam o treino mais lento."
                    ),
                ),
                ParamSpec(
                    "max_depth",
                    "Profundidade máxima",
                    "int",
                    15,
                    2,
                    40,
                    1,
                    help=(
                        "Quantas perguntas cada árvore pode fazer. Ex.: com 2, todas as árvores ficam simples "
                        "e votam quase igual; com 30, cada uma aprende detalhes, e a votação compensa parte "
                        "dos erros individuais."
                    ),
                ),
                ParamSpec(
                    "min_samples_leaf",
                    "Mínimo de restaurantes por folha",
                    "int",
                    2,
                    1,
                    20,
                    1,
                    help=(
                        "Quantos restaurantes, no mínimo, precisam chegar ao fim de cada caminho de cada árvore. "
                        "Ex.: com 1, uma árvore pode decidir com base em um único restaurante; com 20, "
                        "cada voto se apoia em pelo menos 20."
                    ),
                ),
            ),
        ),
        ModelSpec(
            key="knn",
            label="KNN (K vizinhos mais próximos)",
            description=(
                "Procura os K restaurantes mais parecidos (mesmo país, preço e serviços parecidos) e conta os "
                "prêmios deles. Ex.: com K=15, se 11 dos vizinhos são estrelados, a chance de estrela é 73%."
            ),
            build=_knn,
            params=(
                ParamSpec(
                    "n_neighbors",
                    "Número de vizinhos (K)",
                    "int",
                    15,
                    1,
                    75,
                    1,
                    help=(
                        "Quantos restaurantes parecidos são consultados. Ex.: com K=5, se 4 são estrelados, a "
                        "chance é 80%; com K=50, a conta inclui restaurantes menos parecidos e a porcentagem "
                        "fica mais suave. Com K=1, a resposta é sempre 0% ou 100%."
                    ),
                ),
                ParamSpec(
                    "weights",
                    "Peso dos vizinhos",
                    "choice",
                    "distance",
                    options=("uniform", "distance"),
                    help=(
                        "Iguais: cada vizinho vale um voto. Pela distância: os mais parecidos valem mais. "
                        "Ex.: se o vizinho quase idêntico é estrelado, ele puxa a chance de estrela para cima "
                        "mais do que um vizinho pouco parecido."
                    ),
                ),
            ),
        ),
        ModelSpec(
            key="logistic_regression",
            label="Regressão Logística",
            description=(
                "Dá pontos para cada característica (ex.: preço \\$\\$\\$\\$ soma muitos pontos para estrela, "
                "preço \\$ tira), soma tudo e converte o total em porcentagem. Mais pontos = mais chance."
            ),
            build=_logistic_regression,
            params=(
                ParamSpec(
                    "C",
                    "Inverso da regularização (C)",
                    "float",
                    1.0,
                    0.01,
                    10.0,
                    0.01,
                    help=(
                        "Controla o tamanho dos pontos. Ex.: com C=0,01 os pontos ficam pequenos e as "
                        "porcentagens ficam próximas umas das outras (modelo cauteloso); com C=10 os pontos "
                        "crescem e as porcentagens ficam extremas, como 95% ou 3%."
                    ),
                ),
            ),
        ),
    )
}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError as error:
        raise ValueError(f"Modelo desconhecido '{key}'. Opções: {', '.join(MODEL_REGISTRY)}") from error
