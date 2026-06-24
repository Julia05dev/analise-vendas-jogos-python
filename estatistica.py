# streamlit run estatistica.py
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    sm = None
    smf = None

st.set_page_config(page_title="Análise de Vendas de Jogos", layout="wide")

REGIOES = {
    "NA_Sales": "América do Norte",
    "EU_Sales": "Europa",
    "JP_Sales": "Japão",
    "Other_Sales": "Outras regiões",
}

COLUNAS_VENDAS = list(REGIOES.keys())
ANO_CORTE_DIGITAL = 2007


# carrega a base e já separa os jogos em dois períodos, por causa do problema das vendas digitais depois de 2008
@st.cache_data
def carregar_dados() -> pd.DataFrame:
    """Carrega e faz o tratamento básico do dataset."""
    caminho_csv = Path(__file__).parent / "vgsales.csv"
    dados = pd.read_csv(caminho_csv)

    dados = dados.dropna(subset=["Year"]).copy()
    dados["Year"] = dados["Year"].astype(int)

    dados = dados.dropna(subset=["Genre"]).copy()

    dados["Periodo_Mercado"] = np.where(
        dados["Year"] <= ANO_CORTE_DIGITAL,
        "Até 2007: base principal",
        "2008 em diante: possível sub-registro digital",
    )

    return dados


def aplicar_filtros(
    dados: pd.DataFrame,
    publisher: str,
    busca_jogo: str,
    intervalo_anos: tuple[int, int],
) -> pd.DataFrame:
    """Aplica os filtros escolhidos pelo usuário."""
    filtrado = dados.copy()

    if publisher != "Todos":
        filtrado = filtrado[filtrado["Publisher"] == publisher]

    if busca_jogo.strip():
        filtrado = filtrado[
            filtrado["Name"].str.contains(busca_jogo, case=False, na=False)
        ]

    filtrado = filtrado[
        (filtrado["Year"] >= intervalo_anos[0])
        & (filtrado["Year"] <= intervalo_anos[1])
    ]

    return filtrado.copy()


# transforma as vendas das regiões em uma única coluna, para conseguir comparar região e gênero na anova
def transformar_para_formato_longo(dados: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma as colunas de vendas regionais em formato longo.

    Antes:
    NA_Sales | EU_Sales | JP_Sales | Other_Sales

    Depois:
    Regiao | Vendas
    """
    longo = dados.melt(
        id_vars=[
            "Name",
            "Platform",
            "Year",
            "Genre",
            "Publisher",
            "Global_Sales",
            "Periodo_Mercado",
        ],
        value_vars=COLUNAS_VENDAS,
        var_name="Regiao",
        value_name="Vendas",
    )

    longo["Regiao"] = longo["Regiao"].replace(REGIOES)

    longo["Log_Vendas"] = np.log1p(longo["Vendas"])

    return longo


# escolhe qual parte da base vai ser usada no teste, principalmente para não misturar sem cuidado os dados depois de 2008
def preparar_base_estatistica(
    dados_filtrados: pd.DataFrame,
    estrategia_digital: str,
) -> pd.DataFrame:
    """
    Define qual base será usada no teste estatístico,
    considerando o problema das vendas digitais a partir de 2008.
    """
    if estrategia_digital == "Usar apenas jogos até 2007 na inferência principal":
        return dados_filtrados[dados_filtrados["Year"] <= ANO_CORTE_DIGITAL].copy()

    if estrategia_digital == "Usar todos os anos, mas separar por período de mercado":
        return dados_filtrados.copy()

    return dados_filtrados.copy()


# executa a anova de dois fatores para testar se gênero, região e a combinação dos dois influenciam as vendas
def executar_anova_duas_vias(dados_base: pd.DataFrame):
    """
    Executa ANOVA de dois fatores:

    Fator 1: Gênero
    Fator 2: Região
    Interação: Gênero x Região

    Modelo:
    log(1 + vendas) ~ gênero + região + gênero:região
    """
    if sm is None or smf is None:
        return None, "Biblioteca statsmodels não instalada. Instale com: pip install statsmodels"

    longo = transformar_para_formato_longo(dados_base)

    longo = longo[longo["Vendas"] >= 0].copy()

    if longo["Genre"].nunique() < 2 or longo["Regiao"].nunique() < 2:
        return (
            None,
            "A base filtrada precisa ter pelo menos 2 gêneros e 2 regiões para executar ANOVA de dois fatores.",
        )

    if len(longo) < 30:
        return (
            None,
            "A base filtrada ficou pequena demais para uma conclusão estatística minimamente confiável.",
        )

    modelo = smf.ols("Log_Vendas ~ C(Genre) * C(Regiao)", data=longo).fit()
    tabela_anova = sm.stats.anova_lm(modelo, typ=2)

    return tabela_anova, None


# interpreta o p-valor usando o limite de 5%, para saber se o resultado é significativo ou não
def interpretar_p_valor(p_valor: float, alfa: float = 0.05) -> str:
    """Interpreta se o p-valor é significativo."""
    if p_valor < alfa:
        return "significativo"
    return "não significativo"


def formatar_p_valor(p_valor: float) -> str:
    """Formata o p-valor para exibição em português."""
    if p_valor < 0.001:
        return "< 0,001"
    return f"{p_valor:.4f}".replace(".", ",")


df = carregar_dados()


st.title("Análise de Vendas de Jogos")

st.markdown("""
## Contextualização

O mercado de jogos eletrônicos possui vendas distribuídas em diferentes regiões do mundo.
Entretanto, comparar visualmente barras de um gráfico não é suficiente para afirmar que as diferenças
observadas são estatisticamente relevantes.

Por isso, além da análise exploratória, este projeto inclui um teste inferencial para responder de forma
mais objetiva à pergunta central.
""")

st.markdown("""
### Pergunta da análise

**Quais jogos e gêneros apresentam maiores vendas por região? As diferenças entre gêneros e regiões são estatisticamente relevantes?**
""")

st.info(
    "Observação metodológica: a partir de 2008, parte relevante do mercado passou a envolver vendas digitais. "
    "Como o dataset usado registra principalmente vendas físicas, a inferência estatística principal deve ser feita "
    "preferencialmente com jogos lançados até 2007. Os anos posteriores continuam disponíveis para exploração, "
    "mas não devem ser tratados como retrato completo do mercado."
)


# filtros

st.sidebar.header("Filtros")

publisher = st.sidebar.selectbox(
    "Escolha a empresa:",
    ["Todos"] + sorted(df["Publisher"].dropna().unique().tolist()),
)

busca_jogo = st.sidebar.text_input(
    "Buscar jogo por palavra-chave:",
    "",
)

intervalo_anos = st.sidebar.slider(
    "Escolha o intervalo de anos:",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max())),
)

medida = st.sidebar.selectbox(
    "Escolha a medida-resumo:",
    ["Média", "Mediana", "Máximo", "Mínimo", "Soma"],
)

estrategia_digital = st.sidebar.radio(
    "Tratamento do problema das vendas digitais:",
    [
        "Usar apenas jogos até 2007 na inferência principal",
        "Usar todos os anos, mas separar por período de mercado",
        "Usar todos os anos sem ajuste — apenas exploração visual",
    ],
    index=0,
)


# aplicação dos filtros

df_filtrado = aplicar_filtros(df, publisher, busca_jogo, intervalo_anos)

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros selecionados.")
    st.stop()


# medidas resumo

if medida == "Média":
    valor_medida = df_filtrado["Global_Sales"].mean()
elif medida == "Mediana":
    valor_medida = df_filtrado["Global_Sales"].median()
elif medida == "Máximo":
    valor_medida = df_filtrado["Global_Sales"].max()
elif medida == "Mínimo":
    valor_medida = df_filtrado["Global_Sales"].min()
else:
    valor_medida = df_filtrado["Global_Sales"].sum()


# mostra no painel qual base está sendo usada no teste, e não só o intervalo geral escolhido no filtro
base_periodo_exibicao = preparar_base_estatistica(df_filtrado, estrategia_digital)

if base_periodo_exibicao.empty:
    periodo_exibido = "Sem dados"
    quantidade_exibida = 0
else:
    periodo_exibido = (
        f"{base_periodo_exibicao['Year'].min()}–"
        f"{base_periodo_exibicao['Year'].max()}"
    )
    quantidade_exibida = len(base_periodo_exibicao)

col1, col2, col3 = st.columns(3)

col1.metric(f"{medida} de vendas globais", f"{valor_medida:.2f} milhões")
col2.metric(
    "Jogos na base do teste",
    f"{quantidade_exibida:,}".replace(",", ".")
)
col3.metric(
    "Período usado no teste",
    periodo_exibido,
)


# tabela dados filtrados

with st.expander("Ver dados filtrados"):
    st.dataframe(df_filtrado)


# top 10 jogos globais

st.subheader("Top 10 jogos mais vendidos globalmente")

top10 = df_filtrado.nlargest(10, "Global_Sales")

fig_top10 = px.bar(
    top10,
    x="Name",
    y="Global_Sales",
    color="Genre",
    title="Top 10 jogos por vendas globais",
    labels={
        "Name": "Jogo",
        "Global_Sales": "Vendas globais (milhões)",
        "Genre": "Gênero",
    },
)

st.plotly_chart(fig_top10, use_container_width=True)


# cria o gráfico principal de vendas por gênero e região, separando os períodos quando essa opção é escolhida
st.subheader("Vendas por gênero divididas por região")

if estrategia_digital == "Usar todos os anos, mas separar por período de mercado":

    # quando separo por período, mostro primeiro a base até 2007, que é a análise mais segura
    df_ate_2007 = df_filtrado[df_filtrado["Year"] <= ANO_CORTE_DIGITAL].copy()

    if df_ate_2007.empty:
        st.warning("Não há jogos até 2007 com os filtros selecionados.")
    else:
        st.markdown("### Gráfico principal — base até 2007")

        vendas_genero_regiao_principal = df_ate_2007.groupby("Genre", as_index=False)[
            ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
        ].sum()

        vendas_genero_regiao_principal = vendas_genero_regiao_principal.sort_values(
            "Global_Sales",
            ascending=True,
        )

        vendas_empilhadas_principal = vendas_genero_regiao_principal.melt(
            id_vars=["Genre", "Global_Sales"],
            value_vars=COLUNAS_VENDAS,
            var_name="Região",
            value_name="Vendas",
        )

        vendas_empilhadas_principal["Região"] = vendas_empilhadas_principal["Região"].replace(REGIOES)

        fig_principal = px.bar(
            vendas_empilhadas_principal,
            x="Vendas",
            y="Genre",
            color="Região",
            orientation="h",
            title="Vendas por gênero e região — base principal até 2007",
            labels={
                "Genre": "Gênero",
                "Vendas": "Vendas em milhões",
                "Região": "Região",
            },
            text="Vendas",
        )

        fig_principal.update_traces(texttemplate="%{x:.1f}", textposition="inside")
        fig_principal.update_layout(barmode="stack")

        st.plotly_chart(fig_principal, use_container_width=True)

    # depois mostro uma comparação complementar para ver como os dados se comportam antes e depois de 2008
    st.markdown("### Comparação complementar — antes e depois de 2008")

    vendas_genero_regiao_periodo = df_filtrado.groupby(
        ["Periodo_Mercado", "Genre"],
        as_index=False
    )[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].sum()

    vendas_empilhadas_periodo = vendas_genero_regiao_periodo.melt(
        id_vars=["Periodo_Mercado", "Genre", "Global_Sales"],
        value_vars=COLUNAS_VENDAS,
        var_name="Região",
        value_name="Vendas",
    )

    vendas_empilhadas_periodo["Região"] = vendas_empilhadas_periodo["Região"].replace(REGIOES)

    fig_periodo = px.bar(
        vendas_empilhadas_periodo,
        x="Vendas",
        y="Genre",
        color="Região",
        orientation="h",
        facet_col="Periodo_Mercado",
        title="Vendas por gênero e região, separadas por período de mercado",
        labels={
            "Genre": "Gênero",
            "Vendas": "Vendas em milhões",
            "Região": "Região",
            "Periodo_Mercado": "Período de mercado",
        },
        text="Vendas",
    )

    fig_periodo.update_traces(texttemplate="%{x:.1f}", textposition="inside")
    fig_periodo.update_layout(barmode="stack")

    st.plotly_chart(fig_periodo, use_container_width=True)

else:
    vendas_genero_regiao = df_filtrado.groupby("Genre", as_index=False)[
        ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    ].sum()

    vendas_genero_regiao = vendas_genero_regiao.sort_values(
        "Global_Sales",
        ascending=True,
    )

    vendas_empilhadas = vendas_genero_regiao.melt(
        id_vars=["Genre", "Global_Sales"],
        value_vars=COLUNAS_VENDAS,
        var_name="Região",
        value_name="Vendas",
    )

    vendas_empilhadas["Região"] = vendas_empilhadas["Região"].replace(REGIOES)

    fig_principal = px.bar(
        vendas_empilhadas,
        x="Vendas",
        y="Genre",
        color="Região",
        orientation="h",
        title="Vendas por gênero com divisão por região",
        labels={
            "Genre": "Gênero",
            "Vendas": "Vendas em milhões",
            "Região": "Região",
        },
        text="Vendas",
    )

    fig_principal.update_traces(texttemplate="%{x:.1f}", textposition="inside")
    fig_principal.update_layout(barmode="stack")

    st.plotly_chart(fig_principal, use_container_width=True)


# tabela resumo por gênero

tabela_resumo = df_filtrado.groupby("Genre", as_index=False)[
    ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
].sum()

tabela_resumo = tabela_resumo.sort_values(
    "Global_Sales",
    ascending=False,
).reset_index(drop=True)

with st.expander("Ver tabela-resumo por gênero"):
    st.dataframe(tabela_resumo, use_container_width=True)


# cria a parte da anova no aplicativo, para a resposta não depender só da leitura visual do gráfico
st.subheader("Teste estatístico: ANOVA de dois fatores")

st.markdown("""
**Método escolhido:** ANOVA de dois fatores, usando como fatores **Gênero** e **Região**.

A variável analisada é a venda regional por jogo. Como as vendas são muito assimétricas e possuem outliers,
o teste usa `log(1 + vendas)` para reduzir a distorção causada por jogos excepcionalmente populares.

O modelo testado é:

`log(1 + vendas) ~ gênero + região + gênero × região`

A interação `gênero × região` é a parte mais importante para a pergunta do trabalho: ela indica se o padrão de vendas dos gêneros muda conforme a região.
""")

base_estatistica = preparar_base_estatistica(df_filtrado, estrategia_digital)


# organiza e mostra o resultado da anova na tela, junto com uma interpretação simples do p-valor
def exibir_resultado_anova(dados_teste: pd.DataFrame, titulo_periodo: str):
    st.markdown(f"### {titulo_periodo}")

    if dados_teste.empty:
        st.warning("Não há dados suficientes para este período.")
        return

    st.caption(
        f"Base usada no teste: {len(dados_teste):,} jogos, período "
        f"{dados_teste['Year'].min()}–{dados_teste['Year'].max()}.".replace(",", ".")
    )

    tabela_anova, erro_anova = executar_anova_duas_vias(dados_teste)

    if erro_anova:
        st.warning(erro_anova)
        return

    tabela_anova_exibicao = tabela_anova.copy()

    tabela_anova_exibicao = tabela_anova_exibicao.rename(index={
        "C(Genre)": "Gênero",
        "C(Regiao)": "Região",
        "C(Genre):C(Regiao)": "Interação Gênero × Região",
        "Residual": "Resíduo",
    })

    tabela_anova_exibicao = tabela_anova_exibicao.rename(columns={
        "sum_sq": "Soma dos quadrados",
        "df": "Graus de liberdade",
        "F": "Estatística F",
        "PR(>F)": "p-valor",
    })

    st.dataframe(tabela_anova_exibicao, use_container_width=True)

    p_genero = tabela_anova.loc["C(Genre)", "PR(>F)"]
    p_regiao = tabela_anova.loc["C(Regiao)", "PR(>F)"]
    p_interacao = tabela_anova.loc["C(Genre):C(Regiao)", "PR(>F)"]

    st.markdown(f"""
**Interpretação do teste, considerando nível de significância de 5%:**

- Efeito de **gênero**: p-valor = **{formatar_p_valor(p_genero)}** → resultado **{interpretar_p_valor(p_genero)}**.
- Efeito de **região**: p-valor = **{formatar_p_valor(p_regiao)}** → resultado **{interpretar_p_valor(p_regiao)}**.
- Interação **gênero × região**: p-valor = **{formatar_p_valor(p_interacao)}** → resultado **{interpretar_p_valor(p_interacao)}**.
""")

    if p_interacao < 0.05:
        st.success(
            "Conclusão estatística: o padrão de vendas por gênero varia de forma estatisticamente significativa entre as regiões."
        )
    else:
        st.warning(
            "Conclusão estatística: neste período, não há evidência suficiente para afirmar que o padrão de vendas por gênero muda entre as regiões."
        )


# se a opção for separar por período, rodo uma anova para cada parte da base
if estrategia_digital == "Usar todos os anos, mas separar por período de mercado":
    st.info(
        "Nesta opção, o teste é executado separadamente para cada período de mercado. "
        "Isso permite comparar a base principal até 2007 com o período posterior, que pode sofrer sub-registro de vendas digitais."
    )

    for periodo, dados_periodo in df_filtrado.groupby("Periodo_Mercado"):
        exibir_resultado_anova(dados_periodo, periodo)

else:
    if base_estatistica.empty:
        st.warning(
            "A base estatística ficou vazia. Isso pode ocorrer se o filtro escolhido selecionar apenas jogos posteriores a 2007 "
            "e a opção de inferência principal estiver limitada ao período até 2007."
        )
    else:
        exibir_resultado_anova(base_estatistica, "Resultado da ANOVA")


# análise específica pós 2008

st.subheader("Tratamento do problema das vendas digitais a partir de 2008")


# resume a diferença entre os períodos para deixar claro o impacto da base depois de 2008
comparacao_periodos = df_filtrado.groupby("Periodo_Mercado", as_index=False).agg(
    Jogos=("Name", "count"),
    Vendas_Globais=("Global_Sales", "sum"),
    Mediana_Global=("Global_Sales", "median"),
    Media_Global=("Global_Sales", "mean"),
)

st.dataframe(comparacao_periodos, use_container_width=True)

st.markdown("""
O dataset não informa corretamente a totalidade das vendas digitais. Por isso, o código **não tenta inventar ou imputar** vendas ausentes.

A estratégia adotada foi metodológica:

1. marcar os registros como **até 2007** ou **2008 em diante**;
2. usar **até 2007 como base principal da inferência estatística**;
3. manter os anos posteriores apenas como análise exploratória ou comparação de sensibilidade.

Isso evita concluir, de forma errada, que um jogo ou gênero vendeu menos apenas porque parte das vendas digitais não aparece no banco.
""")


# conclusão

st.subheader("Conclusões descritivas com os filtros atuais")

genero_global = tabela_resumo.iloc[0]
genero_na = tabela_resumo.sort_values("NA_Sales", ascending=False).iloc[0]
genero_eu = tabela_resumo.sort_values("EU_Sales", ascending=False).iloc[0]
genero_jp = tabela_resumo.sort_values("JP_Sales", ascending=False).iloc[0]
genero_other = tabela_resumo.sort_values("Other_Sales", ascending=False).iloc[0]

st.markdown(f"""
Com base nos dados filtrados:

- O gênero com maior venda global é **{genero_global['Genre']}**, com aproximadamente **{genero_global['Global_Sales']:.2f} milhões** em vendas.
- Na **América do Norte**, o gênero mais vendido é **{genero_na['Genre']}**, com aproximadamente **{genero_na['NA_Sales']:.2f} milhões**.
- Na **Europa**, o gênero mais vendido é **{genero_eu['Genre']}**, com aproximadamente **{genero_eu['EU_Sales']:.2f} milhões**.
- No **Japão**, o gênero mais vendido é **{genero_jp['Genre']}**, com aproximadamente **{genero_jp['JP_Sales']:.2f} milhões**.
- Em **outras regiões**, o gênero mais vendido é **{genero_other['Genre']}**, com aproximadamente **{genero_other['Other_Sales']:.2f} milhões**.

A parte descritiva mostra **quem vendeu mais**. A ANOVA de dois fatores mostra se essas diferenças têm suporte estatístico.
""")