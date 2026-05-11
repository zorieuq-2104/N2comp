import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Avaliação de Imóveis", layout="wide")

st.title("🏠 Avaliação Inteligente de Imóveis")

# =========================
# LOCALIDADES
# =========================
localidades = {
    "Goiânia": {
        "Centro": 600000,
        "Setor Bueno": 900000,
        "Jardim Goiás": 1100000,
        "Setor Oeste": 1000000,
        "Jardim América": 1300000,
        "Alphaville": 2000000
    },
    "Aparecida de Goiânia": {
        "Buriti Sereno": 400000,
        "Garavelo": 380000,
        "Cidade Vera Cruz": 350000
    },
    "Senador Canedo": {
        "Jardim das Oliveiras": 450000,
        "Residencial Aracy": 420000
    },
    "Trindade": {
        "Centro": 350000,
        "Setor Cristina": 320000
    },
    "Goianira": {
        "Centro": 300000,
        "Jardim Imperial": 320000
    }
}

# =========================
# CIDADE / BAIRRO
# =========================
col1, col2 = st.columns(2)

with col1:
    cidade = st.selectbox("Cidade", list(localidades.keys()))

with col2:
    bairro = st.selectbox("Bairro", list(localidades[cidade].keys()))

# =========================
# GRUPOS
# =========================
st.subheader("⚙️ Grupo de variáveis")

grupo = st.selectbox("Escolha o grupo", [
    "Estrutura",
    "Condomínio",
    "Interior",
    "Localização"
])

dados_entrada = {}

# =========================
# INPUTS POR GRUPO
# =========================

if grupo == "Estrutura":
    dados_entrada["area"] = st.slider("Área (m²)", 40, 200)
    dados_entrada["quartos"] = st.slider("Quartos", 1, 5)
    dados_entrada["banheiros"] = st.slider("Banheiros", 1, 4)
    dados_entrada["vagas"] = st.slider("Vagas", 0, 3)
    dados_entrada["suites"] = st.slider("Suítes", 0, 3)
    dados_entrada["andar"] = st.slider("Andar", 0, 30)
    dados_entrada["idade"] = st.slider("Idade do imóvel", 0, 50)

elif grupo == "Condomínio":
    dados_entrada["condominio"] = st.slider("Condomínio (R$)", 200, 1500)
    dados_entrada["elevador"] = int(st.checkbox("Elevador"))
    dados_entrada["portaria"] = int(st.checkbox("Portaria"))
    dados_entrada["piscina"] = int(st.checkbox("Piscina"))
    dados_entrada["academia"] = int(st.checkbox("Academia"))
    dados_entrada["churrasqueira"] = int(st.checkbox("Churrasqueira"))
    dados_entrada["varanda"] = int(st.checkbox("Varanda"))

elif grupo == "Interior":
    dados_entrada["mobiliado"] = int(st.checkbox("Mobiliado"))
    dados_entrada["ar_condicionado"] = int(st.checkbox("Ar-condicionado"))
    dados_entrada["reformado"] = int(st.checkbox("Reformado"))

elif grupo == "Localização":
    dados_entrada["perto_metro"] = int(st.checkbox("Perto de transporte"))
    dados_entrada["vista_livre"] = int(st.checkbox("Vista livre"))
    dados_entrada["dist_supermercado"] = st.slider("Distância supermercado (km)", 0.1, 5.0)
    dados_entrada["dist_posto"] = st.slider("Distância posto (km)", 0.1, 5.0)
    dados_entrada["dist_hospital"] = st.slider("Distância hospital (km)", 0.1, 10.0)

# =========================
# MODELO
# =========================

@st.cache_data
def gerar_modelo():
    np.random.seed(42)
    dados = []

    for _ in range(1500):
        cidade_r = np.random.choice(list(localidades.keys()))
        bairro_r = np.random.choice(list(localidades[cidade_r].keys()))
        base_preco = localidades[cidade_r][bairro_r]

        area = np.random.randint(40, 200)
        quartos = np.random.randint(1, 5)
        banheiros = np.random.randint(1, 4)
        vagas = np.random.randint(0, 3)
        suites = np.random.randint(0, 3)
        andar = np.random.randint(0, 30)
        idade = np.random.randint(0, 50)

        condominio = np.random.randint(200, 1500)
        elevador = np.random.choice([0,1])
        portaria = np.random.choice([0,1])
        piscina = np.random.choice([0,1])
        academia = np.random.choice([0,1])
        churrasqueira = np.random.choice([0,1])
        varanda = np.random.choice([0,1])

        mobiliado = np.random.choice([0,1])
        ar = np.random.choice([0,1])
        reformado = np.random.choice([0,1])

        metro = np.random.choice([0,1])
        vista = np.random.choice([0,1])

        d1 = np.random.uniform(0.1, 5)
        d2 = np.random.uniform(0.1, 5)
        d3 = np.random.uniform(0.1, 10)

        preco = area * (base_preco / 100)
        preco += quartos * 15000
        preco += banheiros * 12000
        preco += vagas * 8000
        preco *= (1 - idade * 0.005)
        preco += andar * 1000

        if piscina: preco += 30000
        if academia: preco += 20000

        dados.append([
            cidade_r, bairro_r,
            area, quartos, banheiros, vagas,
            condominio, suites, andar, idade,
            elevador, portaria, piscina, academia, churrasqueira, varanda,
            mobiliado, ar, reformado,
            metro, vista,
            d1, d2, d3,
            preco
        ])

    df = pd.DataFrame(dados, columns=[
        "cidade","bairro","area","quartos","banheiros","vagas",
        "condominio","suites","andar","idade",
        "elevador","portaria","piscina","academia","churrasqueira","varanda",
        "mobiliado","ar_condicionado","reformado",
        "perto_metro","vista_livre",
        "dist_supermercado","dist_posto","dist_hospital",
        "preco"
    ])

    df = pd.get_dummies(df, columns=["cidade","bairro"])

    X = df.drop("preco", axis=1)
    y = df["preco"]

    modelo = RandomForestRegressor(n_estimators=200)
    modelo.fit(X, y)

    return modelo, X.columns

modelo, colunas = gerar_modelo()

# =========================
# PREVISÃO
# =========================

if st.button("💰 Calcular Preço"):
    entrada = pd.DataFrame(columns=colunas)
    entrada.loc[0] = 0

    for chave, valor in dados_entrada.items():
        entrada[chave] = valor

    entrada[f"cidade_{cidade}"] = 1
    entrada[f"bairro_{bairro}"] = 1

    preco = modelo.predict(entrada)[0]

    st.success(f"💰 Preço estimado: R$ {preco:,.0f}")
    st.info(f"📉 Mín: R$ {preco*0.9:,.0f} | 📈 Máx: R$ {preco*1.1:,.0f}")
