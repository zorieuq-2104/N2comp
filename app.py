
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Avaliação Inteligente", layout="wide")

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

@st.cache_data
def treinar():
    np.random.seed(42)
    dados = []

    for _ in range(1200):
        cidade = np.random.choice(list(localidades.keys()))
        bairro = np.random.choice(list(localidades[cidade].keys()))
        base = localidades[cidade][bairro]

        area = np.random.randint(40, 200)
        quartos = np.random.randint(1, 5)

        preco = area * (base/100)
        preco += quartos * 15000
        preco *= np.random.normal(1, 0.1)

        dados.append([cidade,bairro,area,quartos,preco])

    df = pd.DataFrame(dados, columns=["cidade","bairro","area","quartos","preco"])
    df = pd.get_dummies(df)

    X = df.drop("preco", axis=1)
    y = df["preco"]

    modelo = RandomForestRegressor(n_estimators=200)
    modelo.fit(X, y)

    return modelo, X.columns

modelo, colunas = treinar()

st.title("🏠 Avaliação Inteligente de Imóveis")

cidade = st.selectbox("Cidade", list(localidades.keys()))
bairro = st.selectbox("Bairro", list(localidades[cidade].keys()))

tab1, tab2 = st.tabs(["🏠 Estrutura", "📍 Localização"])

with tab1:
    area = st.slider("Área", 40, 200, 80)
    quartos = st.slider("Quartos", 1, 5, 2)

with tab2:
    pass

if st.button("Prever preço"):
    entrada = pd.DataFrame(columns=colunas)
    entrada.loc[0] = 0

    entrada["area"] = area
    entrada["quartos"] = quartos
    entrada[f"cidade_{cidade}"] = 1
    entrada[f"bairro_{bairro}"] = 1

    preco = modelo.predict(entrada)[0]

    st.success(f"💰 R$ {preco:,.0f}")
