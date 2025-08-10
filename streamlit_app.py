import streamlit as st
import pandas as pd
import numpy as np
import os

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bundles/Aurora – Panel de Canastas de Mercado", layout="wide")

# --- Aurora theme CSS ---
st.markdown(
    '''
    <style>
    :root {
        --brand1: #5A6FF0;
        --brand2: #8FD3FE;
        --bg: #ffffff;
        --card: #F7F8FB;
        --text: #1C2230;
        --muted: #667085;
        --border: #EAECF0;
    }
    .stApp { background: var(--bg); }
    .block-container { max-width: 1200px; padding-top: 0.5rem; }
    h1, h2, h3, h4 { color: var(--text); }
    p, label, span { color: var(--muted); }
    .aurora-hero {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--border);
        margin: 8px 0 18px 0;
    }
    button[role="tab"] {
        border-radius: 999px !important;
        border: 1px solid var(--border) !important;
        padding: 0.45rem 0.9rem !important;
        color: var(--muted) !important;
        background: #fff !important;
        margin-right: 6px !important;
    }
    button[role="tab"][aria-selected="true"] {
        color: #fff !important;
        background: linear-gradient(135deg,var(--brand1),var(--brand2)) !important;
        border-color: transparent !important;
    }
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Banner
if os.path.exists("5859A6E0-8CBA-4C00-8611-03612A8A3EC6.png"):
    st.markdown('<div class="aurora-hero">', unsafe_allow_html=True)
    st.image("5859A6E0-8CBA-4C00-8611-03612A8A3EC6.png", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.title("Bundles / Panel de Canastas de Mercado")
st.caption("Aplicación interactiva en español, con formato profesional Aurora.")

# Sidebar inputs
st.sidebar.header("Carga y Configuración")
uploaded = st.sidebar.file_uploader("Sube un CSV de transacciones", type=["csv"])
min_support = st.sidebar.slider("Soporte mínimo", 0.001, 0.2, 0.02, 0.001)
min_confidence = st.sidebar.slider("Confianza mínima", 0.1, 1.0, 0.3, 0.05)

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Datos cargados correctamente.")
    st.dataframe(df.head())

    # Dummy tabs (logic placeholders)
    tab_resumen, tab_itemsets, tab_reglas, tab_visuales, tab_descargas = st.tabs(
        ["Resumen", "Itemsets", "Reglas", "Visuales", "Descargas"]
    )
    with tab_resumen:
        st.write("Resumen de datos y métricas clave aquí.")
    with tab_itemsets:
        st.write("Vista de itemsets frecuentes.")
    with tab_reglas:
        st.write("Explorador de reglas de asociación.")
    with tab_visuales:
        st.write("Visualizaciones gráficas.")
    with tab_descargas:
        st.write("Botones de descarga.")
else:
    st.info("Sube un archivo CSV para comenzar.")
