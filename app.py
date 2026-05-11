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
# SESSION STATE (NÃO RESETA VALORES)
# =========================
if "valores" not in st.session_state:
    st.session_state.valores = {
        "area": 100,
        "quartos": 2,
        "banheiros": 2,
        "vagas": 1,
        "suites": 1,
        "andar": 5,
        "idade": 10,
        "condominio": 500,
        "elevador": 1,
        "portaria": 1,
        "piscina": 0,
        "academia": 0,
        "churrasqueira": 0,
        "varanda": 1,
        "mobiliado": 0,
        "ar_condicionado": 0,
        "reformado": 0,
        "perto_metro": 0,
        "vista_livre": 0,
        "dist_supermercado": 2,
        "dist_posto": 2,
        "dist_hospital": 5
    }

valores = st.session_state.valores

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

# =========================
# INPUTS POR GRUPO
# =========================

if grupo == "Estrutura":
    valores["area"] = st.slider("Área (m²)", 40, 200, valores["area"])
    valores["quartos"] = st.slider("Quartos", 1, 5, valores["quartos"])
    valores["banheiros"] = st.slider("Banheiros", 1, 4, valores["banheiros"])
    valores["vagas"] = st.slider("Vagas", 0, 3, valores["vagas"])
    valores["suites"] = st.slider("Suítes", 0, 3, valores["suites"])
    valores["andar"] = st.slider("Andar", 0, 30, valores["andar"])
    valores["idade"] = st.slider("Idade do imóvel", 0, 50, valores["idade"])

elif grupo == "Condomínio":
    valores["condominio"] = st.slider("Condomínio (R$)", 200, 1500, valores["condominio"])
    valores["elevador"] = int(st.checkbox("Elevador", bool(valores["elevador"])))
    valores["portaria"] = int(st.checkbox("Portaria", bool(valores["portaria"])))
    valores["piscina"] = int(st.checkbox("Piscina", bool(valores["piscina"])))
    valores["academia"] = int(st.checkbox("Academia", bool(valores["academia"])))
    valores["churrasqueira"] = int(st.checkbox("Churrasqueira", bool(valores["churrasqueira"])))
    valores["varanda"] = int(st.checkbox("Varanda", bool(valores["varanda"])))

elif grupo == "Interior":
    valores["mobiliado"] = int(st.checkbox("Mobiliado", bool(valores["mobiliado"])))
    valores["ar_condicionado"] = int(st.checkbox("Ar-condicionado", bool(valores["ar_condicionado"])))
    valores["reformado"] = int(st.checkbox("Reformado", bool(valores["reformado"])))

elif grupo == "Localização":
    valores["perto_metro"] = int(st.checkbox("Perto de transporte", bool(valores["perto_metro"])))
    valores["vista_livre"] = int(st.checkbox("Vista livre", bool(valores["vista_livre"])))

    valores["dist_supermercado"] = st.slider(
        "Distância supermercado (km)", 0.1, 5.0, float(valores["dist_supermercado"])
    )

    valores["dist_posto"] = st.slider(
        "Distância posto (km)", 0.1, 5.0, float(valores["dist_posto"])
    )

    valores["dist_hospital"] = st.slider(
        "Distância hospital (km)", 0.1, 10.0, float(valores["dist_hospital"])
    )

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

    for k, v in valores.items():
        entrada[k] = v

    entrada[f"cidade_{cidade}"] = 1
    entrada[f"bairro_{bairro}"] = 1

    preco = modelo.predict(entrada)[0]

    st.success(f"💰 Preço estimado: R$ {preco:,.0f}")
    st.info(f"📉 Mín: R$ {preco*0.9:,.0f} | 📈 Máx: R$ {preco*1.1:,.0f}")
