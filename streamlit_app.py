
import os
import streamlit as st
import pandas as pd
import numpy as np

# Charts: matplotlib only
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# mlxtend for Apriori
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except Exception as e:
    pass

# ----------------------------------------
# Config & Theme (Aurora)
# ----------------------------------------
st.set_page_config(page_title="Bundles/Aurora – Panel de Canastas de Mercado", layout="wide")

# --- Aurora professional theme ---
st.markdown(
    """
    <style>
    :root {
        --brand1: #5A6FF0;   /* indigo-blue */
        --brand2: #8FD3FE;   /* sky gradient end */
        --brand3: #A88BEB;   /* soft violet */
        --bg: #ffffff;
        --card: #F7F8FB;
        --text: #1C2230;
        --muted: #667085;
        --border: #EAECF0;
    }
    .stApp { background: var(--bg); }
    .block-container { max-width: 1200px; padding-top: 0.5rem; }
    h1, h2, h3, h4 { color: var(--text); letter-spacing: 0.2px; }
    p, label, span { color: var(--muted); }
    .aurora-hero {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--border);
        box-shadow: 0 2px 6px rgba(16,24,40,.04);
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
        border-color: transparent !important;
        background: linear-gradient(135deg,var(--brand1),var(--brand2)) !important;
        color: #fff !important;
        box-shadow: 0 2px 6px rgba(90,111,240,.25);
    }
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        border: 1px solid var(--border);
        padding: 0.55rem 0.9rem;
        background: #fff;
        color: var(--text);
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        border-color: var(--brand1);
        box-shadow: 0 2px 6px rgba(90,111,240,.25);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#fbfcff, #f7f9ff);
        border-right: 1px solid var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Bundles / Panel de Canastas de Mercado")
st.caption("Aplicación interactiva generada desde tu notebook. Sube un CSV de transacciones y explora itemsets frecuentes y reglas de asociación.")

# Branded header banner
_banner_path = "5859A6E0-8CBA-4C00-8611-03612A8A3EC6.png"
if os.path.exists(_banner_path):
    st.markdown('<div class="aurora-hero">', unsafe_allow_html=True)
    st.image(_banner_path, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------
# Sidebar controls
# ----------------------------------------
st.sidebar.header("Carga y Configuración")
uploaded = st.sidebar.file_uploader("Sube un CSV de transacciones", type=["csv"])
example_schema = st.sidebar.checkbox("Mostrar esquema esperado / consejos", value=False)

if example_schema:
    st.sidebar.markdown(
        """
**Columnas mínimas (no sensibles a mayúsculas):**
- ID de orden (ej. `order_id`, `OrderNumber`)
- Texto de producto/variante (ej. `variant_title`, `product_name`)

**Opcionales:**
- `quantity`, `price`, `date`
        """
    )

min_support = st.sidebar.slider("Soporte mínimo (fracción de órdenes con el itemset)", 0.001, 0.2, 0.02, 0.001)
min_confidence = st.sidebar.slider("Confianza mínima", 0.1, 1.0, 0.3, 0.05)
min_lift = st.sidebar.slider("Lift mínimo", 0.5, 10.0, 1.0, 0.1)
top_n_products = st.sidebar.slider("Top N productos por frecuencia", 5, 100, 20, 5)

# ----------------------------------------
# Helpers
# ----------------------------------------
def try_detect_columns(df: pd.DataFrame):
    order_col = None
    variant_col = None
    for col in df.columns:
        cl = col.lower()
        if order_col is None and ('order' in cl or 'transaction' in cl or cl.endswith('_id')):
            order_col = col
        if variant_col is None and any(k in cl for k in ['variant', 'product', 'item', 'sku', 'name', 'title']):
            variant_col = col
    return order_col, variant_col

def run_pipeline(df, min_support, min_confidence, min_lift, top_n_products):
    # Detect columns
    order_col, variant_col = try_detect_columns(df)
    if order_col is None or variant_col is None:
        return None, "No se pudieron detectar las columnas. Renombra las columnas de ID de orden y producto."

    # Baskets
    grouped = df.groupby(order_col)[variant_col].apply(lambda s: [str(x) for x in s.dropna().tolist()]).tolist()

    # Transaction encoding
    try:
        te = TransactionEncoder()
        te_ary = te.fit_transform(grouped)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    except Exception as e:
        return None, f"Falló la codificación de transacciones: {e}"

    # Apriori and rules
    try:
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            return None, "No hay itemsets frecuentes con este soporte. Prueba bajando el soporte mínimo."
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
        if min_lift is not None:
            rules = rules[rules['lift'] >= min_lift]
    except Exception as e:
        return None, f"Falló Apriori/reglas: {e}"

    # Product frequency
    prod_freq = pd.Series([p for basket in grouped for p in basket]).value_counts().reset_index()
    prod_freq.columns = ['product', 'count']
    prod_freq['share'] = (prod_freq['count'] / prod_freq['count'].sum()).round(4)
    top_freq = prod_freq.head(top_n_products)

    outputs = {
        "frequent_itemsets": frequent_itemsets.sort_values('support', ascending=False).reset_index(drop=True),
        "rules": rules.sort_values(['lift', 'confidence', 'support'], ascending=False).reset_index(drop=True),
        "product_frequency": prod_freq,
        "top_frequency": top_freq,
        "df_encoded_shape": df_encoded.shape,
        "detected_columns": (order_col, variant_col),
        "baskets": df.groupby(order_col)[variant_col].apply(lambda s: [str(x) for x in s.dropna().tolist()])
    }
    return outputs, None

def _set_to_text(s):
    try:
        return " + ".join(sorted(list(s)))
    except Exception:
        return str(s)

def _spanish_rules(df_rules):
    if df_rules is None or df_rules.empty:
        return df_rules
    out = df_rules.copy()
    if "antecedents" in out.columns:
        out["Antecedente"] = out["antecedents"].apply(_set_to_text)
    if "consequents" in out.columns:
        out["Consecuente"] = out["consequents"].apply(_set_to_text)
    for c in ["support","confidence","lift"]:
        if c in out.columns:
            out[c] = out[c].astype(float).round(4)
    rename_map = {"support":"Soporte","confidence":"Confianza","lift":"Lift"}
    out = out.rename(columns=rename_map)
    cols = [c for c in ["Antecedente","Consecuente","Soporte","Confianza","Lift"] if c in out.columns]
    rest = [c for c in out.columns if c not in cols and c not in ["antecedents","consequents"]]
    return out[cols + rest]

def _spanish_itemsets(df_itemsets):
    if df_itemsets is None or df_itemsets.empty:
        return df_itemsets
    out = df_itemsets.copy()
    if "itemsets" in out.columns:
        out["Itemset"] = out["itemsets"].apply(_set_to_text)
    if "support" in out.columns:
        out["Soporte"] = out["support"].astype(float).round(4)
    keep = [c for c in ["Itemset","Soporte"] if c in out.columns]
    rest = [c for c in out.columns if c not in keep and c not in ["itemsets","support"]]
    return out[keep + rest]

# ----------------------------------------
# Main
# ----------------------------------------
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"No se pudo leer el CSV: {e}")
        df = None

    if df is not None:
        st.subheader("Datos crudos (primeras 200 filas)")
        st.dataframe(df.head(200), use_container_width=True)
        with st.expander("Columnas detectadas", expanded=True):
            oc, vc = try_detect_columns(df)
            st.write({"ID de orden": oc, "Producto/Variante": vc})

        with st.spinner("Ejecutando Apriori..."):
            outputs, err = run_pipeline(df, min_support, min_confidence, min_lift, top_n_products)

        if err:
            st.error(err)
        else:
            st.success("Análisis completo.")

            # Metrics
            oc, vc = outputs['detected_columns']
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Canastas", len(df.groupby(oc)))
            with c2:
                st.metric("Productos únicos", df[vc].nunique())
            with c3:
                st.metric("Tamaño codificado", f"{outputs['df_encoded_shape'][0]} × {outputs['df_encoded_shape'][1]}")

            # Tabs
            tab_resumen, tab_itemsets, tab_reglas, tab_visuales, tab_descargas = st.tabs(
                ["Resumen", "Itemsets", "Reglas", "Visuales", "Descargas"]
            )

            # Resumen
            with tab_resumen:
                st.subheader("Productos más frecuentes (Top 10)")
                top10_products = outputs["product_frequency"].head(10).copy()
                st.dataframe(top10_products.rename(columns={"product":"Producto","count":"Conteo","share":"Participación"}), use_container_width=True)
                if plt is not None and not top10_products.empty:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.barh(range(len(top10_products.iloc[::-1])), top10_products["count"].iloc[::-1])
                    ax.set_yticks(range(len(top10_products.iloc[::-1])))
                    ax.set_yticklabels(top10_products["product"].iloc[::-1])
                    ax.set_xlabel("Conteo")
                    ax.set_title("Top 10 productos por frecuencia")
                    st.pyplot(fig)

                st.subheader("Top 10 reglas por Lift")
                top10_rules = outputs["rules"].sort_values(["lift","confidence","support"], ascending=False).head(10)
                st.dataframe(_spanish_rules(top10_rules), use_container_width=True)

            # Itemsets
            with tab_itemsets:
                st.subheader("Itemsets frecuentes")
                st.dataframe(_spanish_itemsets(outputs["frequent_itemsets"]), use_container_width=True)

                st.markdown("---")
                st.subheader("Productos más fuertes (itemsets de 1)")
                fi = outputs["frequent_itemsets"].copy()
                if not fi.empty:
                    singles = fi[fi["itemsets"].apply(lambda s: len(s)==1)].sort_values("support", ascending=False).head(10)
                    st.dataframe(_spanish_itemsets(singles), use_container_width=True)

                st.markdown("---")
                st.subheader("Pares frecuentes (Top 10)")
                if not fi.empty:
                    pairs = fi[fi["itemsets"].apply(lambda s: len(s)==2)].sort_values("support", ascending=False).head(10)
                    st.dataframe(_spanish_itemsets(pairs), use_container_width=True)

            # Reglas
            with tab_reglas:
                st.subheader("Explorar reglas")
                rules = outputs["rules"].copy()
                if rules.empty:
                    st.info("No hay reglas para mostrar.")
                else:
                    colf1, colf2, colf3, colf4 = st.columns(4)
                    with colf1:
                        min_sup = st.number_input("Soporte mínimo", 0.0, 1.0, float(rules["support"].min()), 0.01, format="%.2f")
                    with colf2:
                        min_conf = st.number_input("Confianza mínima", 0.0, 1.0, 0.3, 0.05, format="%.2f")
                    with colf3:
                        min_lift_f = st.number_input("Lift mínimo", 0.0, 100.0, 1.0, 0.1)
                    with colf4:
                        topn_rules = st.number_input("Mostrar Top N", 10, 500, 100, 10)

                    def _contains_text(sets_series, text):
                        if not text:
                            return pd.Series([True]*len(sets_series))
                        text = text.lower()
                        return sets_series.apply(lambda s: any(text in str(x).lower() for x in list(s)))

                    colf5, colf6 = st.columns(2)
                    with colf5:
                        ant_text = st.text_input("Filtrar antecedente contiene", "")
                    with colf6:
                        con_text = st.text_input("Filtrar consecuente contiene", "")

                    rules_f = rules[(rules["support"]>=min_sup)&(rules["confidence"]>=min_conf)&(rules["lift"]>=min_lift_f)]
                    rules_f = rules_f[_contains_text(rules_f["antecedents"], ant_text)]
                    rules_f = rules_f[_contains_text(rules_f["consequents"], con_text)]
                    rules_f = rules_f.sort_values(["lift","confidence","support"], ascending=False).head(int(topn_rules))
                    st.dataframe(_spanish_rules(rules_f), use_container_width=True)

                    st.markdown("---")
                    st.subheader("Oportunidades (Lift alto • Soporte bajo)")
                    colA, colB, colC = st.columns(3)
                    with colA:
                        opp_min_lift = st.number_input("Lift mínimo (Oportunidades)", 1.0, 50.0, 2.0, 0.1)
                    with colB:
                        opp_max_support = st.number_input("Soporte máximo", 0.0, float(rules["support"].max()), 0.03, 0.005, format="%.3f")
                    with colC:
                        opp_topn = st.number_input("Mostrar Top N", 5, 100, 20, 5)
                    oportunidades = rules[(rules["lift"]>=opp_min_lift)&(rules["support"]<=opp_max_support)]                        .sort_values(["lift","confidence","support"], ascending=False).head(int(opp_topn))
                    st.dataframe(_spanish_rules(oportunidades), use_container_width=True)

            # Visuales
            with tab_visuales:
                st.header("Visuales")
                rules = outputs["rules"].copy()
                fi = outputs["frequent_itemsets"].copy()
                pf = outputs["product_frequency"].copy()
                baskets = outputs["baskets"]
                if plt is None:
                    st.warning("Se requiere matplotlib para las gráficas.")
                else:
                    # 1) Lift Matrix Heatmap (simple imshow)
                    st.subheader("Matriz de Lift producto-a-producto (Antecedente → Consecuente)")
                    K = st.slider("¿Cuántos productos incluir en el mapa de calor?", 5, 40, 10, 1, key="liftK")
                    def _flatten_sets(series_sets):
                        out = []
                        for s in series_sets:
                            out.extend(list(s))
                        return out
                    if not rules.empty:
                        prod_in_rules = pd.Series(_flatten_sets(rules["antecedents"]))                            .append(pd.Series(_flatten_sets(rules["consequents"])), ignore_index=True)
                        top_for_matrix = prod_in_rules.value_counts().head(K).index.tolist()
                    else:
                        top_for_matrix = pf["product"].head(K).tolist()
                    lift_matrix = pd.DataFrame(index=top_for_matrix, columns=top_for_matrix, dtype=float)
                    lift_matrix[:] = np.nan
                    if not rules.empty:
                        for _, rrow in rules.iterrows():
                            for a in list(rrow["antecedents"]):
                                for c in list(rrow["consequents"]):
                                    if a in lift_matrix.index and c in lift_matrix.columns:
                                        lift_matrix.loc[a, c] = rrow["lift"]
                    fig1 = plt.figure()
                    ax = fig1.add_subplot(111)
                    im = ax.imshow(lift_matrix.values, aspect="auto")
                    ax.set_xticks(range(len(top_for_matrix)))
                    ax.set_xticklabels(top_for_matrix, rotation=60, ha="right")
                    ax.set_yticks(range(len(top_for_matrix)))
                    ax.set_yticklabels(top_for_matrix)
                    ax.set_xlabel("Productos consecuentes")
                    ax.set_ylabel("Productos antecedentes")
                    ax.set_title("Matriz de Lift producto-a-producto (Antecedente → Consecuente)")
                    plt.colorbar(im, ax=ax, label="Lift")
                    st.pyplot(fig1)

                    # 2) Rules: Support vs Confidence (color = Lift)
                    st.subheader("Reglas: Soporte vs Confianza (color = Lift)")
                    if not rules.empty:
                        fig2 = plt.figure()
                        ax2 = fig2.add_subplot(111)
                        sc = ax2.scatter(rules["support"], rules["confidence"], c=rules["lift"])
                        ax2.set_xlabel("Soporte")
                        ax2.set_ylabel("Confianza")
                        ax2.set_title("Soporte vs Confianza (color = Lift)")
                        plt.colorbar(sc, ax=ax2, label="Lift")
                        st.pyplot(fig2)

                    # 3) Distribution of Lift Values
                    st.subheader("Distribución de valores de Lift")
                    if not rules.empty:
                        fig3 = plt.figure()
                        ax3 = fig3.add_subplot(111)
                        ax3.hist(rules["lift"].dropna(), bins=30)
                        ax3.set_xlabel("Lift")
                        ax3.set_ylabel("Frecuencia")
                        ax3.set_title("Distribución de Lift")
                        st.pyplot(fig3)

                    # 4) Frequent Itemset Sizes
                    st.subheader("Tamaños de itemsets frecuentes")
                    if not fi.empty:
                        sizes = fi["itemsets"].apply(lambda s: len(s))
                        fig4 = plt.figure()
                        ax4 = fig4.add_subplot(111)
                        ax4.hist(sizes, bins=range(1, sizes.max()+2))
                        ax4.set_xlabel("Tamaño del itemset")
                        ax4.set_ylabel("Conteo")
                        ax4.set_title("Distribución de tamaños de itemsets frecuentes")
                        st.pyplot(fig4)

                        # 5) Support distribution for itemsets
                        fig5 = plt.figure()
                        ax5 = fig5.add_subplot(111)
                        ax5.hist(fi["support"], bins=30)
                        ax5.set_xlabel("Soporte")
                        ax5.set_ylabel("Número de itemsets")
                        ax5.set_title("Distribución del soporte (itemsets frecuentes)")
                        st.pyplot(fig5)

                    # 6) Confidence vs Lift (color = Support)
                    st.subheader("Confianza vs Lift (color = Soporte)")
                    if not rules.empty:
                        fig7 = plt.figure()
                        ax7 = fig7.add_subplot(111)
                        sc2 = ax7.scatter(rules["confidence"], rules["lift"], c=rules["support"])
                        ax7.set_xlabel("Confianza")
                        ax7.set_ylabel("Lift")
                        ax7.set_title("Confianza vs Lift (color = Soporte)")
                        plt.colorbar(sc2, ax=ax7, label="Soporte")
                        st.pyplot(fig7)

                    # 7) Basket Size Histogram & Box Plot
                    st.subheader("Distribución del tamaño de canasta")
                    sizes_b = baskets.apply(len).values
                    fig8 = plt.figure()
                    ax8 = fig8.add_subplot(111)
                    ax8.hist(sizes_b, bins=30)
                    ax8.set_xlabel("Artículos por canasta")
                    ax8.set_ylabel("Frecuencia")
                    ax8.set_title("Histograma del tamaño de canasta")
                    st.pyplot(fig8)

                    fig9 = plt.figure()
                    ax9 = fig9.add_subplot(111)
                    ax9.boxplot(sizes_b, vert=True)
                    ax9.set_ylabel("Artículos por canasta")
                    ax9.set_title("Diagrama de caja del tamaño de canasta")
                    st.pyplot(fig9)

                    # 8) Product Frequency Pareto (Top 20)
                    st.subheader("Frecuencia de producto (Análisis de Pareto)")
                    topN = 20
                    pf_top = pf.head(topN).copy()
                    pf_top["cum_share"] = pf_top["count"].cumsum() / pf_top["count"].sum() * 100.0
                    fig10 = plt.figure()
                    ax10 = fig10.add_subplot(111)
                    ax10.bar(range(len(pf_top)), pf_top["count"])
                    ax10.set_xticks(range(len(pf_top)))
                    ax10.set_xticklabels(pf_top["product"], rotation=60, ha="right")
                    ax10.set_ylabel("Frecuencia")
                    ax10.set_title("Top 20 productos por frecuencia (barras) con % acumulado (línea)")
                    ax10_2 = ax10.twinx()
                    ax10_2.plot(range(len(pf_top)), pf_top["cum_share"])
                    ax10_2.set_ylabel("Porcentaje acumulado")
                    st.pyplot(fig10)

            # Descargas
            with tab_descargas:
                st.subheader("Exportar resultados")
                def df_to_csv_bytes(df_):
                    return df_.to_csv(index=False).encode("utf-8")
                st.download_button("itemsets_frecuentes.csv", data=df_to_csv_bytes(outputs["frequent_itemsets"]), file_name="itemsets_frecuentes.csv")
                st.download_button("reglas_asociacion.csv", data=df_to_csv_bytes(outputs["rules"]), file_name="reglas_asociacion.csv")
                st.download_button("frecuencia_productos.csv", data=df_to_csv_bytes(outputs["product_frequency"]), file_name="frecuencia_productos.csv")
else:
    st.info("Sube un CSV para comenzar. Si quieres que fije el conjunto de datos/visuales de tu notebook, comparte el CSV de muestra (p.ej., Apriori_data.csv).")
