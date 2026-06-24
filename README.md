# Análise de Vendas de Jogos

Projeto desenvolvido em Python com Streamlit para análise exploratória e inferencial de vendas de jogos eletrônicos a partir do dataset `vgsales.csv`.

O objetivo principal é investigar quais jogos e gêneros apresentam maiores vendas por região e verificar, por meio de teste estatístico, se as diferenças entre gêneros e regiões são estatisticamente relevantes.

---

## Objetivo da análise

A pergunta central do projeto é:

> Quais jogos e gêneros apresentam maiores vendas por região? As diferenças entre gêneros e regiões são estatisticamente relevantes?

Para responder a essa pergunta, o projeto combina:

- análise exploratória dos dados;
- visualizações interativas;
- filtros por empresa, jogo e período;
- comparação de vendas por gênero e região;
- teste estatístico ANOVA de dois fatores.

---

## Tecnologias utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Plotly Express
- Statsmodels

---

## Funcionalidades

O aplicativo permite:

- carregar e tratar automaticamente o dataset `vgsales.csv`;
- filtrar os dados por:
  - publisher;
  - nome do jogo;
  - intervalo de anos;
  - medida-resumo;
- visualizar os 10 jogos mais vendidos globalmente;
- comparar vendas por gênero e região;
- consultar uma tabela-resumo por gênero;
- executar ANOVA de dois fatores;
- interpretar automaticamente os p-valores;
- tratar separadamente o problema das vendas digitais a partir de 2008.

---

## Sobre o tratamento dos dados

O projeto considera uma limitação importante do dataset: a partir de 2008, parte relevante do mercado de jogos passou a envolver vendas digitais.

Como o dataset registra principalmente vendas físicas, o aplicativo separa os dados em dois períodos:

- até 2007: base principal da inferência estatística;
- 2008 em diante: possível sub-registro de vendas digitais.

Essa separação evita conclusões incorretas, como afirmar que determinado jogo ou gênero vendeu menos apenas porque parte das vendas digitais não aparece na base.

---

## Teste estatístico utilizado

O projeto utiliza ANOVA de dois fatores para avaliar se há diferenças estatisticamente relevantes nas vendas considerando:

- gênero do jogo;
- região de venda;
- interação entre gênero e região.

O modelo utilizado é:

```text
log(1 + vendas) ~ gênero + região + gênero × região
