"""Display labels for everything shown on screen.

The data keeps its original English values (they come from the dataset and
are what the models learn from). Translation happens only when a value is
displayed, through the helpers below.
"""

from __future__ import annotations

from collections.abc import Iterable

from michelin.config import (
    COL_ADDRESS,
    COL_AWARD,
    COL_CITY,
    COL_COUNTRY,
    COL_CUISINE,
    COL_FACILITIES,
    COL_FACILITY_COUNT,
    COL_GREEN_STAR,
    COL_IS_STARRED,
    COL_LATITUDE,
    COL_LOCATION,
    COL_LONGITUDE,
    COL_NAME,
    COL_PRICE,
    COL_PRICE_LEVEL,
)

AWARD_LABELS = {
    "Selected Restaurants": "Selecionado",
    "Bib Gourmand": "Bib Gourmand",
    "1 Star": "1 Estrela",
    "2 Stars": "2 Estrelas",
    "3 Stars": "3 Estrelas",
    "Starred": "Estrelado",
    "Not starred": "Não estrelado",
}

FACILITY_LABELS = {
    "Air conditioning": "Ar-condicionado",
    "Terrace": "Terraço",
    "Car park": "Estacionamento",
    "Wheelchair access": "Acesso para cadeirantes",
    "Pets not allowed": "Animais não permitidos",
    "Interesting wine list": "Carta de vinhos interessante",
    "Pets allowed everywhere": "Animais permitidos em todo o local",
    "Counter dining": "Refeição no balcão",
    "Pets allowed only in outdoor dining areas (patio/terrace)": "Animais só na área externa",
    "Great view": "Vista privilegiada",
    "Garden or park": "Jardim ou parque",
    "Valet parking": "Manobrista",
    "Cash only": "Somente dinheiro",
    "Brunch": "Brunch",
    "Notable sake list": "Carta de saquês notável",
    "Credit cards not accepted": "Não aceita cartão de crédito",
    "No alcohol served": "Não serve bebida alcoólica",
    "Foreign credit cards not accepted": "Não aceita cartão estrangeiro",
    "Shoes must be removed": "É preciso tirar os sapatos",
    "Bring your own bottle": "Traga sua própria bebida",
    "Cash only - lunch": "Somente dinheiro no almoço",
    "Only service animals permitted": "Apenas animais de serviço",
}

COUNTRY_LABELS = {
    "Andorra": "Andorra",
    "Argentina": "Argentina",
    "Australia": "Austrália",
    "Austria": "Áustria",
    "Belgium": "Bélgica",
    "Brazil": "Brasil",
    "Canada": "Canadá",
    "Chinese Mainland": "China continental",
    "Croatia": "Croácia",
    "Czechia": "Tchéquia",
    "Denmark": "Dinamarca",
    "Estonia": "Estônia",
    "Finland": "Finlândia",
    "France": "França",
    "Germany": "Alemanha",
    "Greece": "Grécia",
    "Hong Kong SAR": "Hong Kong",
    "Hungary": "Hungria",
    "Iceland": "Islândia",
    "Ireland": "Irlanda",
    "Italy": "Itália",
    "Japan": "Japão",
    "Latvia": "Letônia",
    "Liechtenstein": "Liechtenstein",
    "Lithuania": "Lituânia",
    "Luxembourg": "Luxemburgo",
    "Macau SAR": "Macau",
    "Malaysia": "Malásia",
    "Malta": "Malta",
    "Mexico": "México",
    "Netherlands": "Países Baixos",
    "New Zealand": "Nova Zelândia",
    "Norway": "Noruega",
    "Poland": "Polônia",
    "Portugal": "Portugal",
    "Principality of Monaco": "Mônaco",
    "Qatar": "Catar",
    "Saudi Arabia": "Arábia Saudita",
    "Serbia": "Sérvia",
    "Singapore": "Singapura",
    "Slovenia": "Eslovênia",
    "South Korea": "Coreia do Sul",
    "Spain": "Espanha",
    "Sweden": "Suécia",
    "Switzerland": "Suíça",
    "Taiwan Region": "Taiwan",
    "Thailand": "Tailândia",
    "The Philippines": "Filipinas",
    "Türkiye": "Turquia",
    "USA": "Estados Unidos",
    "United Arab Emirates": "Emirados Árabes Unidos",
    "United Kingdom": "Reino Unido",
    "Vietnam": "Vietnã",
}

COLUMN_LABELS = {
    COL_NAME: "Nome",
    COL_ADDRESS: "Endereço",
    COL_LOCATION: "Localização",
    COL_PRICE: "Preço",
    COL_CUISINE: "Culinária",
    COL_LONGITUDE: "Longitude",
    COL_LATITUDE: "Latitude",
    COL_AWARD: "Prêmio",
    COL_GREEN_STAR: "Estrela Verde",
    COL_FACILITIES: "Serviços",
    COL_CITY: "Cidade",
    COL_COUNTRY: "País",
    COL_PRICE_LEVEL: "Nível de preço",
    COL_FACILITY_COUNT: "Qtd. de serviços",
    COL_IS_STARRED: "Estrelado",
    "PhoneNumber": "Telefone",
    "Url": "Página no Guia",
    "WebsiteUrl": "Site",
    "Description": "Descrição",
}

# Hyperparameter choices shown in select boxes.
CHOICE_LABELS = {
    "gini": "Gini",
    "entropy": "Entropia",
    "uniform": "Iguais",
    "distance": "Pela distância",
}

# Feature-name prefixes produced by the model pipeline.
_FEATURE_PREFIXES = {
    "Has: ": "Tem: ",
    f"{COL_COUNTRY}_": "País: ",
    f"{COL_CUISINE}_": "Culinária: ",
}
_INFREQUENT_SUFFIX = "infrequent_sklearn"


def award_label(value: object) -> str:
    """Display name of an award or prediction class."""
    return AWARD_LABELS.get(str(value), str(value))


def award_labels(values: Iterable[object]) -> list[str]:
    return [award_label(value) for value in values]


def country_label(value: str) -> str:
    """Display name of a country; unknown countries keep their original name."""
    return COUNTRY_LABELS.get(value, value)


def facility_label(value: str) -> str:
    return FACILITY_LABELS.get(value, value)


def column_label(value: str) -> str:
    return COLUMN_LABELS.get(value, value)


def choice_label(value: object) -> str:
    return CHOICE_LABELS.get(str(value), str(value))


def feature_label(name: str) -> str:
    """Readable name for a model input column (e.g. "Country_France" -> "País: France")."""
    for prefix, translated in _FEATURE_PREFIXES.items():
        if name.startswith(prefix):
            value = name[len(prefix) :]
            if value == _INFREQUENT_SUFFIX:
                return f"{translated}outros (pouco frequentes)"
            if prefix == "Has: ":
                value = facility_label(value)
            elif prefix == f"{COL_COUNTRY}_":
                value = country_label(value)
            return translated + value
    return column_label(name)


def format_int(value: float) -> str:
    """Integer with Brazilian thousands separator: 19985 -> "19.985"."""
    return f"{value:,.0f}".replace(",", ".")


def format_decimal(value: float, digits: int = 3) -> str:
    """Decimal with comma: 0.725 -> "0,725"."""
    return f"{value:.{digits}f}".replace(".", ",")


def format_percent(value: float, digits: int = 0) -> str:
    """Share as percentage: 0.214 -> "21%"."""
    return f"{value:.{digits}%}".replace(".", ",")
