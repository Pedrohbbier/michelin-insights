# Michelin Insights

Aplicação web para explorar o **Guia Michelin** e prever o prêmio de um restaurante.

Faça upload de **CSV, JSON ou XML**, limpe os dados, explore gráficos e mapas, e treine ou retreine classificadores quando chegar base nova.

Stack: **Streamlit, pandas, matplotlib, seaborn, Plotly, scikit-learn**.

## Como rodar

Python **3.11+**.

```bash
cd michelin-insights
make install
make run
```

Abre em <http://localhost:8501>. A base completa em `data/` carrega automaticamente.

| Comando | O que faz |
|---|---|
| `make install` | Cria o `.venv` e instala as dependências |
| `make install-dev` | O mesmo, mais pytest e ruff |
| `make run` | Sobe o app |
| `make test` | Roda os testes |
| `make lint` / `make format` | Checa ou formata com ruff |
| `make help` | Lista todos os comandos |

## Como usar

Quatro páginas na barra lateral:

| Página | O que faz |
|---|---|
| **Dados** | Upload (substituir ou adicionar), relatório de limpeza e inspeção da tabela |
| **Explorar** | Filtros por país / prêmio / preço; gráficos, mapas e tabela para baixar |
| **Treinar** | Escolhe o classificador, ajusta parâmetros, treina ou retreina e lê as métricas |
| **Prever** | Descreve um restaurante e vê as probabilidades de cada prêmio |

Arquivos de exemplo por país ficam em `data/samples/`, úteis para testar upload e retreino.

## Dataset

Fonte: [Michelin Guide Restaurants](https://www.kaggle.com/datasets/ngshiheng/michelin-guide-restaurants-2021) no Kaggle ([ngshiheng/michelin-my-maps](https://github.com/ngshiheng/michelin-my-maps)).

Licença: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Cópia no projeto: `data/michelin_restaurants.csv`.

Colunas principais: `Name`, `Address`, `Location`, `Price`, `Cuisine`, `Longitude`, `Latitude`, `Award`, `GreenStar`, `FacilitiesAndServices`.

Distribuição dos prêmios na base incluída:

| Selected | Bib Gourmand | 1 Star | 2 Stars | 3 Stars |
|---:|---:|---:|---:|---:|
| 12,276 | 3,802 | 3,200 | 546 | 161 |

## Limpeza

`michelin/data/cleaning.py` valida o schema, limpa textos, remove duplicatas, normaliza prêmios e preços, valida coordenadas, preenche lacunas e cria colunas derivadas (`City`, `Country`, `FacilityList`, `IsStarred`, …). As remoções aparecem no relatório de limpeza.

## Machine learning

Classificação supervisionada do prêmio Michelin (Selected / Bib Gourmand / Estrelado) a partir de país, culinária, preço, serviços e Estrela Verde.

Classificadores: Árvore de Decisão, Random Forest, KNN e Regressão Logística, cada um com seus parâmetros. O treino usa um `Pipeline` do scikit-learn, split estratificado 80/20, acurácia no teste e matriz de confusão. Dá para retreinar depois de mudar os dados ativos.

## Estrutura do projeto

```
michelin-insights/
├── streamlit_app.py
├── app/                      # Interface Streamlit
│   ├── state.py
│   ├── components.py
│   ├── descriptions.py
│   └── views/                # Dados, Explorar, Treinar, Prever
├── michelin/                 # Lógica (sem Streamlit)
│   ├── config.py
│   ├── labels.py
│   ├── data/                 # carga, schema, limpeza
│   ├── analysis/             # agregações
│   ├── viz/                  # gráficos e mapas
│   └── ml/                   # features, modelos, treino, previsão
├── data/
│   ├── michelin_restaurants.csv
│   └── samples/
├── models/
├── scripts/
└── tests/
```

## Testes

```bash
make install-dev
make test
make lint
```
