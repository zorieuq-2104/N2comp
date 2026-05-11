import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Avaliação de Imóveis", layout="wide")

st.title("🏠 Avaliação Inteligente de Imóveis (Profissional)")

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
# INPUTS
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    cidade = st.selectbox("Cidade", list(localidades.keys()))
    bairro = st.selectbox("Bairro", list(localidades[cidade].keys()))
    area = st.slider("Área (m²)", 40, 200)
    quartos = st.slider("Quartos", 1, 5)
    banheiros = st.slider("Banheiros", 1, 4)

with col2:
    vagas = st.slider("Vagas", 0, 3)
    suites = st.slider("Suítes", 0, 3)
    andar = st.slider("Andar", 0, 30)
    idade = st.slider("Idade do imóvel", 0, 50)
    condominio = st.slider("Condomínio (R$)", 200, 1500)

with col3:
    piscina = st.checkbox("Piscina")
    academia = st.checkbox("Academia")
    elevador = st.checkbox("Elevador")
    portaria = st.checkbox("Portaria")
    churrasqueira = st.checkbox("Churrasqueira")
    varanda = st.checkbox("Varanda")
    mobiliado = st.checkbox("Mobiliado")
    ar_condicionado = st.checkbox("Ar-condicionado")
    reformado = st.checkbox("Reformado")
    perto_metro = st.checkbox("Perto de transporte")
    vista_livre = st.checkbox("Vista livre")

# Distâncias
st.subheader("📍 Localização")
dist_supermercado = st.slider("Distância ao supermercado (km)", 0.1, 5.0)
dist_posto = st.slider("Distância ao posto (km)", 0.1, 5.0)
dist_hospital = st.slider("Distância ao hospital (km)", 0.1, 10.0)

# =========================
# GERAR DATASET E TREINAR
# =========================

@st.cache_data
def gerar_modelo():
    np.random.seed(42)
    dados = []

    for _ in range(1500):
        cidade_r = np.random.choice(list(localidades.keys()))
        bairro_r = np.random.choice(list(localidades[cidade_r].keys()))
        base_preco = localidades[cidade_r][bairro_r]

        area_r = np.random.randint(40, 200)
        quartos_r = np.random.randint(1, 5)
        banheiros_r = np.random.randint(1, 4)
        vagas_r = np.random.randint(0, 3)
        suites_r = np.random.randint(0, 3)
        andar_r = np.random.randint(0, 30)
        idade_r = np.random.randint(0, 50)

        condominio_r = np.random.randint(200, 1500)
        elevador_r = np.random.choice([0,1])
        portaria_r = np.random.choice([0,1])
        piscina_r = np.random.choice([0,1])
        academia_r = np.random.choice([0,1])
        churrasqueira_r = np.random.choice([0,1])
        varanda_r = np.random.choice([0,1])

        mobiliado_r = np.random.choice([0,1])
        ar_r = np.random.choice([0,1])
        reformado_r = np.random.choice([0,1])

        metro_r = np.random.choice([0,1])
        vista_r = np.random.choice([0,1])

        d1 = np.random.uniform(0.1, 5)
        d2 = np.random.uniform(0.1, 5)
        d3 = np.random.uniform(0.1, 10)

        preco = area_r * (base_preco/100)
        preco += quartos_r * 15000
        preco += banheiros_r * 12000
        preco += vagas_r * 8000
        preco *= (1 - idade_r * 0.005)
        preco += andar_r * 1000

        if piscina_r: preco += 30000
        if academia_r: preco += 20000

        dados.append([
            cidade_r, bairro_r,
            area_r, quartos_r, banheiros_r, vagas_r,
            condominio_r, suites_r, andar_r, idade_r,
            elevador_r, portaria_r, piscina_r, academia_r, churrasqueira_r, varanda_r,
            mobiliado_r, ar_r, reformado_r,
            metro_r, vista_r,
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

    entrada["area"] = area
    entrada["quartos"] = quartos
    entrada["banheiros"] = banheiros
    entrada["vagas"] = vagas
    entrada["condominio"] = condominio
    entrada["suites"] = suites
    entrada["andar"] = andar
    entrada["idade"] = idade

    entrada["piscina"] = int(piscina)
    entrada["academia"] = int(academia)
    entrada["elevador"] = int(elevador)
    entrada["portaria"] = int(portaria)
    entrada["churrasqueira"] = int(churrasqueira)
    entrada["varanda"] = int(varanda)
    entrada["mobiliado"] = int(mobiliado)
    entrada["ar_condicionado"] = int(ar_condicionado)
    entrada["reformado"] = int(reformado)
    entrada["perto_metro"] = int(perto_metro)
    entrada["vista_livre"] = int(vista_livre)

    entrada["dist_supermercado"] = dist_supermercado
    entrada["dist_posto"] = dist_posto
    entrada["dist_hospital"] = dist_hospital

    entrada[f"cidade_{cidade}"] = 1
    entrada[f"bairro_{bairro}"] = 1

    preco = modelo.predict(entrada)[0]

    st.success(f"💰 Preço estimado: R$ {preco:,.0f}")
    st.info(f"📉 Mín: R$ {preco*0.9:,.0f} | 📈 Máx: R$ {preco*1.1:,.0f}")
