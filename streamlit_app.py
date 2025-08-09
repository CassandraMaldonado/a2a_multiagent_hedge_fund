
import streamlit as st
import pandas as pd
import numpy as np

# Try to import optional libs used in the notebook
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
sns = None  # enforce matplotlib-only for charts

# mlxtend for apriori
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except Exception as e:
    st.warning("mlxtend is required for Apriori analysis. Install with: pip install mlxtend")

st.set_page_config(page_title="Bundles/Aurora – Market Basket Dashboard", layout="wide")

st.title("🧺 Bundles / Market Basket Dashboard")
st.caption("Interactive app generated from your notebook. Upload a transactions CSV and explore frequent itemsets & association rules.")

st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
example_schema = st.sidebar.checkbox("Show expected schema / tips", value=False)

if example_schema:
    st.sidebar.markdown(
        '''
**Minimum columns (case-insensitive, app will try to auto-detect):**
- An order id column (e.g., `order_id`, `OrderNumber`)
- A product/variant text column (e.g., `variant_title`, `product_name`)

**Optional helpful columns:**
- `quantity`, `price`, `date`
        '''
    )

min_support = st.sidebar.slider("Min support (fraction of orders containing an itemset)", 0.001, 0.2, 0.02, 0.001)
min_confidence = st.sidebar.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
min_lift = st.sidebar.slider("Min lift", 0.5, 10.0, 1.0, 0.1)
top_n_products = st.sidebar.slider("Top N products by frequency", 5, 100, 20, 5)

# ---- Notebook code (functions/utilities) ----
# Below is the code extracted from the uploaded notebook. 
# It will be available to the app for analysis.

# >>> BEGIN NOTEBOOK CODE >>>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def load_and_group_transactions(csv_file):

    df = pd.read_csv('/content/Apriori_data.csv')

    df.columns = df.columns.str.strip().str.lower()

    print(f"Column names found: {list(df.columns)}")

    order_col = None
    variant_col = None

    for col in df.columns:
        if 'order' in col.lower() and 'id' in col.lower():
            order_col = col
            break

    for col in df.columns:
        if 'variant' in col.lower() or 'product' in col.lower() or 'name' in col.lower():
            variant_col = col
            break

    if order_col is None:
        possible_order_cols = ['order_id', 'orderid', 'order', 'id']
        for col in possible_order_cols:
            if col in df.columns:
                order_col = col
                break

    if variant_col is None:
        possible_variant_cols = ['variant_name', 'variantname', 'product_name', 'productname', 'name', 'product']
        for col in possible_variant_cols:
            if col in df.columns:
                variant_col = col
                break

    if order_col is None or variant_col is None:
        print(f"Could not find the columns.")
        print(f"Available columns: {list(df.columns)}")
        print(f"Looking for order column (found: {order_col}) and variant/product column (found: {variant_col})")
        raise ValueError("Required columns not found.")

    print(f"Using columns: '{order_col}' and '{variant_col}'")

    # Quitando nulls.
    df = df.dropna(subset=[order_col, variant_col])

    print(f"Loaded {len(df)} transaction records")
    print(f"Unique orders: {df[order_col].nunique()}")
    print(f"Unique products: {df[variant_col].nunique()}")

    # Productos por order id.
    transactions = df.groupby(order_col)[variant_col].apply(list).tolist()

    print(f"Created {len(transactions)} transaction baskets.")
    print(f"Average basket size: {np.mean([len(t) for t in transactions]):.2f} items")

    print("\nSample transactions:")
    for i, transaction in enumerate(transactions[:3]):
        print(f"  Order {i+1}: {transaction[:3]}{'...' if len(transaction) > 3 else ''}")

    return transactions, df

def prepare_data_for_apriori(transactions):

    # TransactionEncoder.
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)

    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    print(f"Encoded data shape: {df_encoded.shape}")
    print(f"Products in analysis: {len(df_encoded.columns)}")

    return df_encoded

def run_apriori_analysis(df_encoded, min_support=0.01, min_lift=1.0):
    print(f"\n Apriori analysis.")
    print(f"Minimum support: {min_support}")
    print(f"Minimum lift: {min_lift}")

    # Items frecuentes.
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    print(f"Found {len(frequent_itemsets)} frequent itemsets.")

    if len(frequent_itemsets) == 0:
        print("No frequent itemsets found.")
        return pd.DataFrame(), pd.DataFrame()

    # Reglas de asociasion.
    try:
        rules = association_rules(frequent_itemsets,
                                metric="lift",
                                min_threshold=min_lift,
                                num_itemsets=len(frequent_itemsets))

        print(f"Generated {len(rules)} association rules.")

        if len(rules) == 0:
            print("No rules.")
            return frequent_itemsets, pd.DataFrame()

        rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

        return frequent_itemsets, rules

    except ValueError as e:
        print(f"Error generating rules: {e}")
        return frequent_itemsets, pd.DataFrame()

def display_top_rules(rules, top_n=10):
    if rules.empty:
        print("No rules to display.")
        return

    print(f"\n{'_'*80}")
    print(f"Top {top_n} Association Rules")
    print(f"{'_'*80}")

    top_rules = rules.head(top_n)

    for i, row in top_rules.iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))

        print(f"\n{i+1:2d}. IF customer buys: {antecedent}")
        print(f"    THEN they also buy: {consequent}")
        print(f"    Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.2f}")


def create_frequent_products_chart(df_original, top_n=10):
    print(f"\n Creating bar chart of top {top_n} most frequent products.")

    variant_col = None
    for col in df_original.columns:
        if 'variant' in col.lower() or 'product' in col.lower() or 'name' in col.lower():
            variant_col = col
            break

    if variant_col is None:
        print("Could not find product/variant column for chart!")
        return

    # Frequencias.
    product_counts = df_original[variant_col].value_counts().head(top_n)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(product_counts)), product_counts.values,
                   color='skyblue', edgecolor='navy', alpha=0.7)

    plt.title(f'Top {top_n} Most Frequent Products', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Products', fontsize=12)
    plt.ylabel('Frequency (Number of Orders)', fontsize=12)

    plt.xticks(range(len(product_counts)),
               [name[:30] + '...' if len(name) > 30 else name for name in product_counts.index],
               rotation=45, ha='right')

    for i, (bar, count) in enumerate(zip(bars, product_counts.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_lift_heatmap(rules, top_n=10):
    if rules.empty:
        print("No rules.")
        return

    print(f"\n Heatmap for top {top_n} product pairs.")

    # Top rules.
    top_rules = rules.head(top_n).copy()

    top_rules['antecedent_str'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    top_rules['consequent_str'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # Heatmap
    rule_labels = []
    lift_values = []

    for _, row in top_rules.iterrows():
        rule_label = f"{row['antecedent_str'][:20]}...\\n→ {row['consequent_str'][:20]}..."
        rule_labels.append(rule_label)
        lift_values.append(row['lift'])

    lift_matrix = np.array(lift_values).reshape(1, -1)

    plt.figure(figsize=(15, 6))

    sns.heatmap(lift_matrix,
                xticklabels=[f"Rule {i+1}" for i in range(len(rule_labels))],
                yticklabels=['Lift Value'],
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Lift Value'},
                square=False)

    plt.title(f'Lift Values for Top {top_n} Association Rules', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Association Rules', fontsize=12)
    plt.ylabel('', fontsize=12)

    # Rule details.
    rule_details = "\\n".join([f"Rule {i+1}: {rule_labels[i].replace(chr(10), ' → ')}"
                              for i in range(min(5, len(rule_labels)))])
    plt.figtext(0.02, 0.02, rule_details, fontsize=8, verticalalignment='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if len(top_rules) >= 2:
        create_detailed_lift_heatmap(top_rules, top_n)

def create_detailed_lift_heatmap(top_rules, top_n):

    all_products = set()
    for _, row in top_rules.iterrows():
        all_products.update(row['antecedents'])
        all_products.update(row['consequents'])

    all_products = sorted(list(all_products))

    if len(all_products) > 15:
        # Productos que aparecen mas frecuentenmente.
        product_frequency = {}
        for _, row in top_rules.iterrows():
            for product in row['antecedents']:
                product_frequency[product] = product_frequency.get(product, 0) + 1
            for product in row['consequents']:
                product_frequency[product] = product_frequency.get(product, 0) + 1

        # Productos top.
        top_products = sorted(product_frequency.keys(),
                            key=lambda x: product_frequency[x], reverse=True)[:15]
        all_products = top_products

    # Lift matrix.
    lift_matrix = pd.DataFrame(0.0, index=all_products, columns=all_products)

    for _, row in top_rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                if antecedent in all_products and consequent in all_products:
                    lift_matrix.loc[antecedent, consequent] = row['lift']

    # Heatmap.
    plt.figure(figsize=(12, 10))

    # Masking zero values.
    mask = lift_matrix == 0

    sns.heatmap(lift_matrix,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                mask=mask,
                square=True,
                cbar_kws={'label': 'Lift Value'},
                linewidths=0.5)

    plt.title('Product-to-Product Lift Matrix\\n(Antecedent → Consequent)',
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Consequent Products', fontsize=12)
    plt.ylabel('Antecedent Products', fontsize=12)

    # Rotate labels for better readability.
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

def create_comprehensive_dashboard(df_original, frequent_itemsets, rules):
    print("-"*80)
    print("Dashboard")
    print("-"*80)

    order_col = None
    variant_col = None

    for col in df_original.columns:
        if 'order' in col.lower():
            order_col = col
        if 'variant' in col.lower() or 'product' in col.lower() or 'name' in col.lower():
            variant_col = col

    if order_col is None or variant_col is None:
        print("Cannot create dashboard - missing required columns")
        return

    fig = plt.figure(figsize=(20, 16))

    # Size de las transaacciones
    plt.subplot(3, 4, 1)
    basket_sizes = df_original.groupby(order_col).size()
    plt.hist(basket_sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Basket Sizes', fontweight='bold')
    plt.xlabel('Number of Items per Basket')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)

    plt.axvline(basket_sizes.mean(), color='red', linestyle='--',
                label=f'Mean: {basket_sizes.mean():.1f}')
    plt.axvline(basket_sizes.median(), color='orange', linestyle='--',
                label=f'Median: {basket_sizes.median():.1f}')
    plt.legend()

    # support vs confidence
    plt.subplot(3, 4, 2)
    if not rules.empty:
        scatter = plt.scatter(rules['support'], rules['confidence'],
                            c=rules['lift'], cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Lift')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Rules: Support vs Confidence\\n(Color = Lift)', fontweight='bold')
        plt.grid(alpha=0.3)

    # productos top
    plt.subplot(3, 4, 3)
    product_counts = df_original[variant_col].value_counts().head(8)

    bars = plt.bar(range(len(product_counts)), product_counts.values, color='lightcoral')
    plt.title('Top Products by Frequency', fontweight='bold')
    plt.xlabel('Products')
    plt.ylabel('Frequency')
    plt.xticks(range(len(product_counts)),
               [name[:10] + '...' if len(name) > 10 else name for name in product_counts.index],
               rotation=45)

    for bar, count in zip(bars, product_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                str(count), ha='center', va='bottom', fontsize=9)

    # Lift
    plt.subplot(3, 4, 4)
    if not rules.empty:
        plt.hist(rules['lift'], bins=15, alpha=0.7, color='gold', edgecolor='black')
        plt.title('Distribution of Lift Values', fontweight='bold')
        plt.xlabel('Lift')
        plt.ylabel('Number of Rules')
        plt.axvline(1.0, color='red', linestyle='--', label='Lift = 1.0')
        plt.legend()
        plt.grid(alpha=0.3)

    # distribución del tamaño del conjunto
    plt.subplot(3, 4, 5)
    if not frequent_itemsets.empty:
        itemset_sizes = frequent_itemsets['itemsets'].apply(len)
        size_counts = itemset_sizes.value_counts().sort_index()

        bars = plt.bar(size_counts.index, size_counts.values, color='lightgreen')
        plt.title('Frequent Itemset Sizes', fontweight='bold')
        plt.xlabel('Itemset Size')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)

        for bar, count in zip(bars, size_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                    str(count), ha='center', va='bottom')

    # support
    plt.subplot(3, 4, 6)
    if not frequent_itemsets.empty:
        plt.hist(frequent_itemsets['support'], bins=15, alpha=0.7,
                color='plum', edgecolor='black')
        plt.title('Support Distribution', fontweight='bold')
        plt.xlabel('Support')
        plt.ylabel('Number of Itemsets')
        plt.grid(alpha=0.3)

    # top antecedentes
    plt.subplot(3, 4, 7)
    if not rules.empty:
        antecedents = []
        for rule in rules['antecedents']:
            antecedents.extend(list(rule))

        antecedent_counts = Counter(antecedents).most_common(8)
        if antecedent_counts:
            products, counts = zip(*antecedent_counts)
            bars = plt.bar(range(len(products)), counts, color='orange', alpha=0.7)
            plt.title('Most Frequent Antecedents', fontweight='bold')
            plt.xlabel('Products')
            plt.ylabel('Frequency in Rules')
            plt.xticks(range(len(products)), [p[:15] + '...' if len(p) > 15 else p
                      for p in products], rotation=45)

    # Consequentes
    plt.subplot(3, 4, 8)
    if not rules.empty:
        consequents = []
        for rule in rules['consequents']:
            consequents.extend(list(rule))

        consequent_counts = Counter(consequents).most_common(8)
        if consequent_counts:
            products, counts = zip(*consequent_counts)
            bars = plt.bar(range(len(products)), counts, color='cyan', alpha=0.7)
            plt.title('Most Frequent Consequents', fontweight='bold')
            plt.xlabel('Products')
            plt.ylabel('Frequency in Rules')
            plt.xticks(range(len(products)), [p[:15] + '...' if len(p) > 15 else p
                      for p in products], rotation=45)

    # confidence vs lift
    plt.subplot(3, 4, 9)
    if not rules.empty:
        scatter = plt.scatter(rules['confidence'], rules['lift'],
                            c=rules['support'], cmap='plasma', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Support')
        plt.xlabel('Confidence')
        plt.ylabel('Lift')
        plt.title('Confidence vs Lift\\n(Color = Support)', fontweight='bold')
        plt.grid(alpha=0.3)

    # distribucion de las ordenes
    plt.subplot(3, 4, 10)
    customer_orders = df_original.groupby(order_col).size()
    plt.boxplot(customer_orders)
    plt.title('Basket Size Distribution\\n(Box Plot)', fontweight='bold')
    plt.ylabel('Items per Basket')
    plt.grid(alpha=0.3)

    # rule quality
    plt.subplot(3, 4, 11)
    if not rules.empty and len(rules) >= 5:
        metrics = ['support', 'confidence', 'lift']
        top_5_rules = rules.head(5)

        x = np.arange(len(top_5_rules))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = top_5_rules[metric].values
            if metric == 'lift':
                values = values / max(values) * max(top_5_rules['confidence'])  # Normalize for visualization

            plt.bar(x + i * width, values, width,
                   label=metric.capitalize(), alpha=0.7)

        plt.title('Top 5 Rules - Quality Metrics', fontweight='bold')
        plt.xlabel('Rule Rank')
        plt.ylabel('Normalized Value')
        plt.xticks(x + width, [f'Rule {i+1}' for i in range(len(top_5_rules))])
        plt.legend()
        plt.grid(alpha=0.3)

    # Frequencia de product pareto
    plt.subplot(3, 4, 12)
    product_counts = df_original[variant_col].value_counts()
    cumulative_percentage = np.cumsum(product_counts) / product_counts.sum() * 100

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    bars = ax1.bar(range(min(20, len(product_counts))),
                   product_counts[:20].values, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Products (Top 20)')
    ax1.set_ylabel('Frequency', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2.plot(range(min(20, len(cumulative_percentage))),
             cumulative_percentage[:20], color='red', marker='o')
    ax2.set_ylabel('Cumulative %', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Product Frequency (Pareto Analysis)', fontweight='bold')

    plt.tight_layout()
    plt.show()

def create_network_graph(rules, min_lift=1.5, top_n=15):
    if rules.empty:
        print("No rules available.")
        return

    # Filter rules by lift
    filtered_rules = rules[rules['lift'] >= min_lift].head(top_n)

    if len(filtered_rules) == 0:
        print(f"No rules found with lift >= {min_lift}")
        return

    # network graph
    G = nx.Graph()

    # edges with weights
    for _, rule in filtered_rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])

        # edges between antecedents and consequents
        for ant in antecedents:
            for cons in consequents:
                if G.has_edge(ant, cons):
                    G[ant][cons]['weight'] += rule['lift']
                    G[ant][cons]['count'] += 1
                else:
                    G.add_edge(ant, cons, weight=rule['lift'], count=1)

    if len(G.nodes()) == 0:
        print("No nodes in graph!")
        return

    plt.figure(figsize=(16, 12))

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Edges con thickness basado en el peso.
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1

    edge_widths = [3 * (w / max_weight) for w in weights]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')

    node_sizes = []
    for node in G.nodes():
        # Tamaño del node basado en el degree.
        degree = G.degree(node)
        node_sizes.append(500 + degree * 300)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color='lightblue', alpha=0.8, edgecolors='navy')


    labels = {node: node[:15] + '...' if len(node) > 15 else node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')

    plt.title(f'Product Relationship Network\\n'
             f'(Min Lift: {min_lift}, Node size = connections, Edge width = lift strength)',
             fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Network created with {len(G.nodes())} products and {len(G.edges())} connections")
    print(f"Average degree: {np.mean([d for n, d in G.degree()]):.2f}")

    # Productos mas conectados.
    degrees = dict(G.degree())
    most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n Most connected products:")
    for product, degree in most_connected:
        print(f"  • {product}: {degree} connections")

def generate_business_insights(rules, df_original, frequent_itemsets):
    print("-"*80)
    print("Analysis")
    print("-"*80)

    order_col = None
    variant_col = None

    for col in df_original.columns:
        if 'order' in col.lower():
            order_col = col
        if 'variant' in col.lower() or 'product' in col.lower() or 'name' in col.lower():
            variant_col = col

    if order_col is None or variant_col is None:
        print("Cannot generate insights.")
        return

    # Stats.
    total_orders = df_original[order_col].nunique()
    total_products = df_original[variant_col].nunique()
    avg_basket_size = df_original.groupby(order_col).size().mean()

    print(f"\n Resumen:")
    print(f"Total de ordenes: {total_orders:,}")
    print(f"Products unicos: {total_products:,}")
    print(f"Promedio de la basket size: {avg_basket_size:.2f} items")
    print(f"Total de productos vendidos: {len(df_original):,}")

    if not rules.empty:
        print(f"\n Reglas de asociacion:")
        print(f"Total de reglas: {len(rules):,}")
        print(f"Lift promedio: {rules['lift'].mean():.2f}")
        print(f"Confianza promedio: {rules['confidence'].mean():.2f}")
        print(f"Mejor lift score: {rules['lift'].max():.2f}")

        # Top reglas.
        high_lift_rules = rules[rules['lift'] > 2.0]
        high_confidence_rules = rules[rules['confidence'] > 0.7]

        print("-"*80)
        print(f"\n Oportunidades:")
        print(f"Reglas con lift > 2.0: {len(high_lift_rules)} (Asociasiones fuertes)")
        print(f"Rules con confianza > 70%: {len(high_confidence_rules)} (Prediciones confiables)")

        # Recomendaciones de productos.
        print(f"\n Recomendaciones de productos:")
        for i, (_, rule) in enumerate(rules.head(5).iterrows()):
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            print(f"  {i+1}. Cuando compran '{antecedent[:30]}{'...' if len(antecedent) > 30 else ''}'")
            print(f"     → recomendar '{consequent[:30]}{'...' if len(consequent) > 30 else ''}' "
                  f"(Confianza: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f})")

    # Pares frequentes.
    if not frequent_itemsets.empty:
        print("-"*80)
        print(f"\n Pares frequentes:")
        itemset_sizes = frequent_itemsets['itemsets'].apply(len)
        print(f"Un producto (tamaño 1): {len(itemset_sizes[itemset_sizes == 1])}")
        print(f"Pares (tamaño 2): {len(itemset_sizes[itemset_sizes == 2])}")
        print(f"Tres productos (tamaño 3): {len(itemset_sizes[itemset_sizes >= 3])}")

        # Los productos mas fuertes de manera individual.
        print("-"*80)
        single_items = frequent_itemsets[itemset_sizes == 1].sort_values('support', ascending=False)
        print(f"\n Los productos mas fuertes de manera individual:")
        for i, (_, item) in enumerate(single_items.head(5).iterrows()):
            product = list(item['itemsets'])[0]
            print(f"  {i+1}. {product[:40]}{'...' if len(product) > 40 else ''} "
                  f"(Support: {item['support']:.1%})")
        print("-"*80)

def export_comprehensive_results(rules, frequent_itemsets, df_original):

    order_col = None
    variant_col = None

    for col in df_original.columns:
        if 'order' in col.lower():
            order_col = col
        if 'variant' in col.lower() or 'product' in col.lower() or 'name' in col.lower():
            variant_col = col

    # reglas de asociacion.
    if not rules.empty:
        export_rules = rules.copy()
        export_rules['antecedents_str'] = export_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        export_rules['consequents_str'] = export_rules['consequents'].apply(lambda x: ', '.join(list(x)))

        export_columns = ['antecedents_str', 'consequents_str', 'antecedent support',
                         'consequent support', 'support', 'confidence', 'lift', 'conviction']

        export_df = export_rules[export_columns].copy()
        export_df.columns = ['Antecedents', 'Consequents', 'Antecedent_Support',
                            'Consequent_Support', 'Support', 'Confidence', 'Lift', 'Conviction']

        export_df.to_csv('apriori_rules_detailed.csv', index=False)
        print(f"Exported {len(export_df)} rules to 'apriori_rules_detailed.csv'")

    # pares frequentes.
    if not frequent_itemsets.empty:
        frequent_export = frequent_itemsets.copy()
        frequent_export['itemsets_str'] = frequent_export['itemsets'].apply(lambda x: ', '.join(list(x)))
        frequent_export['itemset_size'] = frequent_export['itemsets'].apply(len)

        frequent_final = frequent_export[['itemsets_str', 'itemset_size', 'support']].copy()
        frequent_final.columns = ['Itemset', 'Size', 'Support']
        frequent_final.to_csv('frequent_itemsets.csv', index=False)
        print(f"Exported {len(frequent_final)} frequent itemsets to 'frequent_itemsets.csv'")

    # analysis de productos.
    if variant_col and order_col:
        product_analysis = df_original[variant_col].value_counts().reset_index()
        product_analysis.columns = ['Product', 'Frequency']
        product_analysis['Support'] = product_analysis['Frequency'] / df_original[order_col].nunique()
        product_analysis.to_csv('product_frequency_analysis.csv', index=False)
        print(f"Exported product analysis to 'product_frequency_analysis.csv'")

        # basket analysis.
        basket_analysis = df_original.groupby(order_col).agg({
            variant_col: ['count', lambda x: ', '.join(x)]
        }).reset_index()
        basket_analysis.columns = ['Order_ID', 'Basket_Size', 'Products']
        basket_analysis.to_csv('basket_analysis.csv', index=False)
        print(f"Exported basket analysis to 'basket_analysis.csv'")

def main():

    csv_file = 'Apriori_data.csv'
    min_support = 0.01
    min_lift = 1.0

    try:
        transactions, df_original = load_and_group_transactions(csv_file)

        df_encoded = prepare_data_for_apriori(transactions)

        # apriori analysis
        frequent_itemsets, rules = run_apriori_analysis(df_encoded, min_support, min_lift)

        # Top 10
        display_top_rules(rules, top_n=10)

        print("-"*80)
        print("Gráficas")
        print("-"*80)

        # Productos frequentes.
        create_frequent_products_chart(df_original, top_n=10)

        # Heatmap de lift.
        create_lift_heatmap(rules, top_n=10)

        create_comprehensive_dashboard(df_original, frequent_itemsets, rules)

        # network analysis.
        create_network_graph(rules, min_lift=1.5, top_n=15)

        generate_business_insights(rules, df_original, frequent_itemsets)

        export_comprehensive_results(rules, frequent_itemsets, df_original)

        return {
            'transactions': transactions,
            'frequent_itemsets': frequent_itemsets,
            'rules': rules,
            'df_original': df_original
        }

    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please make sure the file exists and update the path in the code.")
        return None

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
# <<< END NOTEBOOK CODE <<<

def try_detect_columns(df):
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
    # Try to locate order and product columns
    order_col, variant_col = try_detect_columns(df)
    if order_col is None or variant_col is None:
        return None, "Could not auto-detect columns. Please rename your order id and product columns."

    # Group into baskets (list of items per order)
    grouped = df.groupby(order_col)[variant_col].apply(lambda s: [str(x) for x in s.dropna().tolist()]).tolist()

    # Transaction encoding
    try:
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit_transform(grouped)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    except Exception as e:
        return None, f"Transaction encoding failed: {e}"

    # Run apriori
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            return None, "No frequent itemsets at this support. Try lowering min_support."
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
        if min_lift is not None:
            rules = rules[rules['lift'] >= min_lift]
    except Exception as e:
        return None, f"Apriori/rules failed: {e}"

    # Product frequency
    prod_freq = pd.Series([p for basket in grouped for p in basket]).value_counts().reset_index()
    prod_freq.columns = ['product', 'count']
    prod_freq['share'] = prod_freq['count'] / prod_freq['count'].sum()
    top_freq = prod_freq.head(top_n_products)

    outputs = {
        "frequent_itemsets": frequent_itemsets.sort_values('support', ascending=False).reset_index(drop=True),
        "rules": rules.sort_values(['lift', 'confidence', 'support'], ascending=False).reset_index(drop=True),
        "product_frequency": prod_freq,
        "top_frequency": top_freq,
        "df_encoded_shape": df_encoded.shape,
        "detected_columns": (order_col, variant_col),
    }
    return outputs, None

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None

    if df is not None:
        st.subheader("📄 Raw Data (first 200 rows)")
        st.dataframe(df.head(200), use_container_width=True)
        with st.expander("Detected columns", expanded=True):
            oc, vc = try_detect_columns(df)
            st.write({"Order ID": oc, "Product/Variant": vc})

        with st.spinner("Running Apriori…"):
            outputs, err = run_pipeline(df, min_support, min_confidence, min_lift, top_n_products)

        if err:
            st.error(err)
        else:
            st.success("Analysis complete.")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Baskets", len(df.groupby(outputs['detected_columns'][0])))
            with c2:
                st.metric("Unique products", df[outputs['detected_columns'][1]].nunique())
            with c3:
                st.metric("Encoded shape", f"{outputs['df_encoded_shape'][0]} × {outputs['df_encoded_shape'][1]}")

            st.subheader("🏆 Top Products by Frequency")
            st.dataframe(outputs["top_frequency"], use_container_width=True)

            if sns is not None and plt is not None:
                st.pyplot(plt.figure())
                plt.figure()
                sns.barplot(data=outputs["top_frequency"], x="count", y="product")
                plt.title("Top Products by Count")
                st.pyplot(plt.gcf())

            st.subheader("📈 Frequent Itemsets")
            # --- Top 10 Association Rules ---
            st.subheader("🔝 Top 10 Association Rules (ranked by Lift, Confidence, Support)")
            top10_rules = outputs["rules"].copy()
            def _set_to_text(s):
                try:
                    return " + ".join(sorted(list(s)))
                except Exception:
                    return str(s)
            if not top10_rules.empty:
                top10_rules = top10_rules.assign(
                    antecedent_text=top10_rules["antecedents"].apply(_set_to_text),
                    consequent_text=top10_rules["consequents"].apply(_set_to_text),
                    rule=lambda d: d["antecedent_text"] + " → " + d["consequent_text"]
                ).sort_values(["lift","confidence","support"], ascending=False).head(10)
                st.dataframe(top10_rules[["rule","support","confidence","lift"]], use_container_width=True)
            else:
                st.info("No rules to display.")

            # --- Top 10 Most Frequent Products ---
            st.subheader("🏅 Top 10 Most Frequent Products")
            top10_products = outputs["product_frequency"].head(10).copy()
            st.dataframe(top10_products, use_container_width=True)
            if sns is not None and plt is not None and not top10_products.empty:
                plt.figure()
                sns.barplot(data=top10_products, x="count", y="product")
                plt.title("Top 10 Products by Frequency")
                st.pyplot(plt.gcf())

            # --- Oportunidades (High Lift, Low Support) ---
            st.subheader("💡 Oportunidades (High Lift • Low Support)")
            colA, colB, colC = st.columns(3)
            with colA:
                opp_min_lift = st.number_input("Min Lift (Oportunidades)", min_value=1.0, max_value=50.0, value=2.0, step=0.1)
            with colB:
                opp_max_support = st.number_input("Max Support", min_value=0.0, max_value=float(outputs["rules"]["support"].max() if not outputs["rules"].empty else 0.5), value=0.03, step=0.005, format="%.3f")
            with colC:
                opp_topn = st.number_input("Show Top N", min_value=5, max_value=100, value=20, step=5)
            oportunidades = outputs["rules"].copy()
            if not oportunidades.empty:
                oportunidades = oportunidades[(oportunidades["lift"] >= opp_min_lift) & (oportunidades["support"] <= opp_max_support)]\
                    .sort_values(["lift","confidence","support"], ascending=False).head(int(opp_topn))
                if not oportunidades.empty:
                    oportunidades = oportunidades.assign(
                        antecedent_text=oportunidades["antecedents"].apply(_set_to_text),
                        consequent_text=oportunidades["consequents"].apply(_set_to_text),
                        rule=lambda d: d["antecedent_text"] + " → " + d["consequent_text"]
                    )
                    st.dataframe(oportunidades[["rule","support","confidence","lift"]], use_container_width=True)
                else:
                    st.info("No opportunities at these thresholds. Try lowering Max Support or Min Lift.")
            else:
                st.info("No rules calculated to derive opportunities.")

            # --- Pares Frecuentes (Top Frequent Pairs) ---
            st.subheader("👫 Pares Frecuentes (Top 10 itemsets of size 2)")
            fi = outputs["frequent_itemsets"].copy()
            if not fi.empty:
                pairs = fi[fi["itemsets"].apply(lambda s: len(s)==2)].sort_values("support", ascending=False).head(10).copy()
                if not pairs.empty:
                    pairs["pair"] = pairs["itemsets"].apply(lambda s: " + ".join(sorted(list(s))))
                    st.dataframe(pairs[["pair","support"]], use_container_width=True)
                else:
                    st.info("No frequent pairs at current support.")
            else:
                st.info("No frequent itemsets yet.")

            # --- Productos más fuertes (1-item itemsets by support) ---
            st.subheader("💪 Productos más fuertes (Top 10 1-item itemsets)")
            if not fi.empty:
                singles = fi[fi["itemsets"].apply(lambda s: len(s)==1)].sort_values("support", ascending=False).head(10).copy()
                if not singles.empty:
                    singles["producto"] = singles["itemsets"].apply(lambda s: next(iter(s)))
                    st.dataframe(singles[["producto","support"]], use_container_width=True)
                else:
                    st.info("No single-item itemsets at this support threshold.")

            # --- Optional visuals / images section ---
            st.subheader("🖼️ Visuals")
            img_files = []
            default_paths = [
                "67A69E2F-D98B-48B5-B15B-E33F0341462B.png",
                "BDB36D4B-CD16-47F4-8E2E-97FAAC3A8ECF.jpeg",
            ]
            for p in default_paths:
                if os.path.exists(p):
                    img_files.append(p)
            uploaded_imgs = st.file_uploader("Add images (optional)", type=["png","jpg","jpeg"], accept_multiple_files=True)
            if uploaded_imgs:
                for uf in uploaded_imgs:
                    img_files.append(uf)
            if img_files:
                for im in img_files:
                    try:
                        st.image(im, use_column_width=True)
                    except Exception:
                        st.image(im.read(), use_column_width=True)
            else:
                st.caption("No images found. Upload one or place it next to the app as the filenames above.")

            st.dataframe(outputs["frequent_itemsets"], use_container_width=True)

            st.subheader("🔗 Association Rules")
            st.dataframe(outputs["rules"], use_container_width=True)

            # Downloads
            st.subheader("⬇️ Downloads")
            def df_to_csv_bytes(df_):
                return df_.to_csv(index=False).encode("utf-8")

            st.download_button("Download frequent_itemsets.csv", data=df_to_csv_bytes(outputs["frequent_itemsets"]), file_name="frequent_itemsets.csv")
            st.download_button("Download association_rules.csv", data=df_to_csv_bytes(outputs["rules"]), file_name="association_rules.csv")
            st.download_button("Download product_frequency.csv", data=df_to_csv_bytes(outputs["product_frequency"]), file_name="product_frequency.csv")
            # =======================
            # Comprehensive Visuals
            # =======================
            st.header("📊 Visuals")

            if plt is None:
                st.warning("matplotlib is required for charts.")
            else:
                oc, vc = outputs['detected_columns']

                # --- Prep basics ---
                # Baskets and basket sizes
                baskets = df.groupby(oc)[vc].apply(lambda s: [str(x) for x in s.dropna().tolist()])
                basket_sizes = baskets.apply(len).values

                # Convenience: top products and rules
                pf = outputs["product_frequency"].copy()
                rules = outputs["rules"].copy()
                fi = outputs["frequent_itemsets"].copy()

                # 1) Lift Matrix Heatmap (Top K products appearing in rules)
                st.subheader("Product-to-Product Lift Matrix (Antecedent → Consequent)")
                K = st.slider("How many products to include in heatmap", 5, 40, 10, 1, key="liftK")
                # find products that appear in rules to improve density
                def _flatten_sets(s):
                    out = []
                    for t in s:
                        out.extend(list(t))
                    return out
                prod_in_rules = pd.Series(_flatten_sets(rules["antecedents"])) if not rules.empty else pd.Series([], dtype=str)
                prod_in_rules = pd.concat([prod_in_rules, pd.Series(_flatten_sets(rules["consequents"]))], ignore_index=True) if not rules.empty else prod_in_rules
                if not prod_in_rules.empty:
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
                ax.set_xlabel("Consequent Products")
                ax.set_ylabel("Antecedent Products")
                ax.set_title("Product-to-Product Lift Matrix (Antecedent → Consequent)")
                plt.colorbar(im, ax=ax, label="Lift")
                st.pyplot(fig1)

                # 2) Rules: Support vs Confidence (color = Lift)
                st.subheader("Rules: Support vs Confidence (color = Lift)")
                if not rules.empty:
                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111)
                    sc = ax2.scatter(rules["support"], rules["confidence"], c=rules["lift"])
                    ax2.set_xlabel("Support")
                    ax2.set_ylabel("Confidence")
                    ax2.set_title("Support vs Confidence (color = Lift)")
                    plt.colorbar(sc, ax=ax2, label="Lift")
                    st.pyplot(fig2)
                else:
                    st.info("No rules to plot.")

                # 3) Distribution of Lift Values
                st.subheader("Distribution of Lift Values")
                if not rules.empty:
                    fig3 = plt.figure()
                    ax3 = fig3.add_subplot(111)
                    ax3.hist(rules["lift"].dropna(), bins=30)
                    ax3.set_xlabel("Lift")
                    ax3.set_ylabel("Frequency")
                    ax3.set_title("Lift Distribution")
                    st.pyplot(fig3)

                # 4) Frequent Itemset Sizes
                st.subheader("Frequent Itemset Sizes")
                if not fi.empty:
                    sizes = fi["itemsets"].apply(lambda s: len(s))
                    fig4 = plt.figure()
                    ax4 = fig4.add_subplot(111)
                    ax4.hist(sizes, bins=range(1, sizes.max()+2))
                    ax4.set_xlabel("Itemset Size")
                    ax4.set_ylabel("Count")
                    ax4.set_title("Distribution of Frequent Itemset Sizes")
                    st.pyplot(fig4)

                    # 5) Support distribution for itemsets
                    fig5 = plt.figure()
                    ax5 = fig5.add_subplot(111)
                    ax5.hist(fi["support"], bins=30)
                    ax5.set_xlabel("Support")
                    ax5.set_ylabel("Number of Itemsets")
                    ax5.set_title("Support Distribution (Frequent Itemsets)")
                    st.pyplot(fig5)

                # 6) Most Frequent Antecedents / Consequents
                st.subheader("Most Frequent Antecedents / Consequents in Rules (Top 10)")
                if not rules.empty:
                    from collections import Counter
                    ant_counts = Counter()
                    con_counts = Counter()
                    for _, rrow in rules.iterrows():
                        ant_counts.update(list(rrow["antecedents"]))
                        con_counts.update(list(rrow["consequents"]))
                    ant_top = pd.Series(ant_counts).sort_values(ascending=False).head(10)
                    con_top = pd.Series(con_counts).sort_values(ascending=False).head(10)
                    fig6a = plt.figure()
                    ax6a = fig6a.add_subplot(111)
                    ax6a.barh(range(len(ant_top.index[::-1])), ant_top.values[::-1])
                    ax6a.set_yticks(range(len(ant_top.index[::-1])))
                    ax6a.set_yticklabels(ant_top.index[::-1])
                    ax6a.set_xlabel("Frequency in Rules")
                    ax6a.set_title("Most Frequent Antecedents")
                    st.pyplot(fig6a)

                    fig6b = plt.figure()
                    ax6b = fig6b.add_subplot(111)
                    ax6b.barh(range(len(con_top.index[::-1])), con_top.values[::-1])
                    ax6b.set_yticks(range(len(con_top.index[::-1])))
                    ax6b.set_yticklabels(con_top.index[::-1])
                    ax6b.set_xlabel("Frequency in Rules")
                    ax6b.set_title("Most Frequent Consequents")
                    st.pyplot(fig6b)

                # 7) Confidence vs Lift (color = Support)
                st.subheader("Confidence vs Lift (color = Support)")
                if not rules.empty:
                    fig7 = plt.figure()
                    ax7 = fig7.add_subplot(111)
                    sc2 = ax7.scatter(rules["confidence"], rules["lift"], c=rules["support"])
                    ax7.set_xlabel("Confidence")
                    ax7.set_ylabel("Lift")
                    ax7.set_title("Confidence vs Lift (color = Support)")
                    plt.colorbar(sc2, ax=ax7, label="Support")
                    st.pyplot(fig7)

                # 8) Basket Size Histogram & Box Plot
                st.subheader("Basket Size Distribution")
                fig8 = plt.figure()
                ax8 = fig8.add_subplot(111)
                ax8.hist(basket_sizes, bins=30)
                ax8.set_xlabel("Items per Basket")
                ax8.set_ylabel("Frequency")
                ax8.set_title("Basket Size Histogram")
                st.pyplot(fig8)

                fig9 = plt.figure()
                ax9 = fig9.add_subplot(111)
                ax9.boxplot(basket_sizes, vert=True)
                ax9.set_ylabel("Items per Basket")
                ax9.set_title("Basket Size Box Plot")
                st.pyplot(fig9)

                # 9) Product Frequency Pareto (Top 20)
                st.subheader("Product Frequency (Pareto Analysis)")
                topN = 20
                pf_top = pf.head(topN).copy()
                pf_top["cum_share"] = pf_top["count"].cumsum() / pf_top["count"].sum() * 100.0
                fig10 = plt.figure()
                ax10 = fig10.add_subplot(111)
                ax10.bar(range(len(pf_top)), pf_top["count"])
                ax10.set_xticks(range(len(pf_top)))
                ax10.set_xticklabels(pf_top["product"], rotation=60, ha="right")
                ax10.set_ylabel("Frequency")
                ax10.set_title("Top 20 Products by Frequency (bars) with Cumulative % (line)")
                ax10_2 = ax10.twinx()
                ax10_2.plot(range(len(pf_top)), pf_top["cum_share"])
                ax10_2.set_ylabel("Cumulative %")
                st.pyplot(fig10)


else:
    st.info("Upload a CSV to begin. If you want me to hardwire your notebook's dataset and visuals, share the sample CSV (e.g., Apriori_data.csv) used in the notebook.")

