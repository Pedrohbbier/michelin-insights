"""Short explanation shown under every chart: what it shows and how to read it.

The texts are Markdown; dollar signs are escaped because Streamlit reads "$" as a math formula.
"""

from __future__ import annotations

CHART_DESCRIPTIONS = {
    "award_bar": (
        "**O que mostra:** quantos restaurantes há em cada categoria de prêmio. "
        "**Como ler:** cada barra é um prêmio, do mais comum (Selecionado) ao mais raro (3 Estrelas). "
        "O número em cima da barra é a quantidade de restaurantes."
    ),
    "award_pie": (
        "**O que mostra:** quanto cada prêmio representa do total de restaurantes filtrados. "
        "**Como ler:** fatias maiores são prêmios mais comuns. As estrelas formam uma fatia pequena, "
        "ou seja, as classes são desbalanceadas, e isso importa na hora de treinar o modelo."
    ),
    "price_heatmap": (
        "**O que mostra:** como os restaurantes de cada faixa de preço (\\$ a \\$\\$\\$\\$) "
        "se dividem entre os prêmios. "
        "**Como ler:** cada linha soma 100%; quanto mais escura a cor, maior a porcentagem. "
        "Exemplo: na linha \\$\\$\\$\\$ dá para ver quantos dos restaurantes mais caros têm estrela."
    ),
    "restaurant_map": (
        "**O que mostra:** onde fica cada restaurante, com a cor indicando o prêmio. "
        "**Como ler:** passe o mouse em um ponto para ver nome, cidade, culinária e preço. "
        "Use a roda do mouse para dar zoom e clique na legenda para esconder ou mostrar um prêmio."
    ),
    "density_map": (
        "**O que mostra:** as regiões com mais restaurantes estrelados (1, 2 ou 3 estrelas). "
        "**Como ler:** quanto mais vermelha e intensa a área, mais restaurantes com estrela existem ali, "
        "próximos uns dos outros."
    ),
    "country_stacked": (
        "**O que mostra:** os 15 países com mais restaurantes no guia e a mistura de prêmios em cada um. "
        "**Como ler:** o comprimento total da barra é o número de restaurantes do país; "
        "cada cor é um prêmio (veja a legenda)."
    ),
    "city_ranking": (
        "**O que mostra:** as 15 cidades com mais restaurantes estrelados. "
        "**Como ler:** quanto maior a barra, mais restaurantes com estrela a cidade tem."
    ),
    "cuisine_ranking": (
        "**O que mostra:** os 15 tipos de culinária mais frequentes (ou só entre os estrelados, se a opção "
        "acima estiver ligada). **Como ler:** o número na ponta da barra é a quantidade de restaurantes."
    ),
    "facility_comparison": (
        "**O que mostra:** a porcentagem de restaurantes estrelados (vermelho) e não estrelados (cinza) "
        "que oferecem cada serviço. **Como ler:** quando a barra vermelha é bem maior que a cinza, "
        "o serviço é mais comum entre os estrelados."
    ),
    "facility_box": (
        "**O que mostra:** quantos serviços os restaurantes listam, separado por prêmio (boxplot). "
        "**Como ler:** a linha dentro da caixa é a mediana; a caixa cobre a metade central dos restaurantes "
        "(de 25% a 75%); os pontos soltos são valores fora do padrão (outliers)."
    ),
    "correlation": (
        "**O que mostra:** a correlação de Pearson entre as variáveis numéricas. "
        "**Como ler:** vai de -1 a 1. Perto de 1 (vermelho), as duas sobem juntas; perto de -1 (azul), "
        "uma sobe e a outra desce; perto de 0, não há relação linear. "
        "'Nível do prêmio' vai de 0 (Selecionado) a 4 (3 Estrelas)."
    ),
    "confusion_matrix": (
        "**O que mostra:** os acertos e erros do modelo nos dados de teste (que ele não viu no treino). "
        "**Como ler:** cada linha é o prêmio real e cada coluna é o que o modelo previu. A diagonal são os "
        "acertos. Fora dela estão os erros: por exemplo, linha 'Estrelado' e coluna 'Selecionado' são "
        "restaurantes com estrela que o modelo classificou como apenas selecionados."
    ),
    "probability": (
        "**O que mostra:** a probabilidade que o modelo dá para cada prêmio, para o restaurante descrito. "
        "**Como ler:** a maior barra é a previsão. Probabilidades parecidas indicam que o modelo está em dúvida."
    ),
    "price_sensitivity": (
        "**O que mostra:** as probabilidades para o mesmo restaurante, mudando só a faixa de preço. "
        "**Como ler:** cada coluna é um preço e cada cor é um prêmio. Mostra o quanto o preço, sozinho, "
        "muda a previsão do modelo."
    ),
}
