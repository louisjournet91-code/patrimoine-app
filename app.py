import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURATION & STYLE "INSTITUTIONNEL" ---
st.set_page_config(
    page_title="Gestion Patrimoniale",
    layout="wide",
    page_icon="🏛️"
)

# CSS : Le style "Feuille Blanche" (Clean & Crisp)
st.markdown("""
<style>
    /* Fond global gris très clair pour faire ressortir les cartes */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Les Cartes (Containers) : Fond blanc, ombre légère, bords arrondis */
    div[data-testid="stMetric"], div.stDataFrame, div.stPlotlyChart {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Titres : Bleu nuit institutionnel, police Serif élégante */
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    
    /* Onglets : Style minimaliste */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 8px;
        color: #64748b;
        border: 1px solid #e2e8f0;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0f172a; /* Actif = Sombre */
        color: #ffffff;
    }
    
</style>
""", unsafe_allow_html=True)

# --- 2. DONNÉES (VOTRE RÉALITÉ) ---
def load_portfolio():
    """Données conformes à votre PDF PEA Marie"""
    data = {
        "Ticker": ["ESE.PA", "CW8.PA", "BTC-EUR", "CASH"],
        "Nom": ["BNP S&P 500", "Amundi MSCI World", "Bitcoin (Réserve)", "Liquidités Espèces"],
        "Type": ["ETF Action", "ETF Action", "Crypto", "Cash"],
        # Vos quantités exactes
        "Quantité": [141, 10, 0.015, 510.84],
        # Vos PRU
        "PRU": [24.41, 400.00, 60000.00, 1.00]
    }
    return pd.DataFrame(data)

@st.cache_data(ttl=600)
def get_prices(df):
    """Récupération silencieuse des prix"""
    tickers = [t for t in df['Ticker'].unique() if t != "CASH"]
    prices = {"CASH": 1.0}
    
    if tickers:
        try:
            data = yf.download(tickers, period="1d", progress=False)['Close']
            # Gestion des formats de retour yfinance (Série vs DataFrame)
            if hasattr(data, 'columns'): # Plusieurs tickers
                last_row = data.iloc[-1]
                for t in tickers:
                    prices[t] = float(last_row[t]) if t in last_row else 0.0
            else: # Un seul ticker
                prices[tickers[0]] = float(data.iloc[-1])
        except:
            pass # En cas d'erreur, on garde 0 ou les anciennes valeurs
            
    return prices

# --- 3. CALCULS ---
df = load_portfolio()
market_prices = get_prices(df)

df['Prix_Actuel'] = df['Ticker'].apply(lambda x: market_prices.get(x, 0.0))
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['Plus_Value'] = df['Valo'] - df['Investi']
df['Perf_%'] = ((df['Prix_Actuel'] - df['PRU']) / df['PRU']) * 100

# Totaux
total_valo = df['Valo'].sum()
total_investi = df['Investi'].sum()
total_pv = total_valo - total_investi
total_perf = (total_pv / total_investi) * 100

# --- 4. INTERFACE UTILISATEUR ---

# En-tête simple et efficace
c_head1, c_head2 = st.columns([3, 1])
with c_head1:
    st.title("Synthèse Patrimoniale")
    st.caption(f"Situation arrêtée au {datetime.now().strftime('%d/%m/%Y à %H:%M')}")

# --- ZONE KPI (Les Gros Chiffres) ---
# On utilise des colonnes avec des métriques natives qui seront stylisées par le CSS
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Patrimoine Net", f"{total_valo:,.2f} €", help="Valeur de liquidation immédiate")
kpi2.metric("Capital Engagé", f"{total_investi:,.2f} €", help="Somme des virements effectués")
kpi3.metric("Plus-Value Latente", f"{total_pv:+,.2f} €", f"{total_perf:+.2f} %")
# Calcul du % de Cash
cash_pct = (df[df['Type']=='Cash']['Valo'].sum() / total_valo) * 100
kpi4.metric("Liquidités Disponibles", f"{df[df['Type']=='Cash']['Valo'].sum():,.2f} €", f"{cash_pct:.1f}% du total")

st.markdown("---")

# --- ZONE PRINCIPALE (Onglets) ---
tab_pf, tab_analytics, tab_simu = st.tabs(["📋 Détail Portefeuille", "📊 Analyse Visuelle", "🔮 Projections"])

# ONGLET 1 : TABLEAU & DONUT
with tab_pf:
    col_table, col_graph = st.columns([2, 1])
    
    with col_table:
        st.subheader("Inventaire des Actifs")
        # Configuration du tableau pour qu'il soit très lisible
        st.dataframe(
            df[['Nom', 'Type', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo', 'Perf_%']],
            column_config={
                "Nom": st.column_config.TextColumn("Actif", width="medium"),
                "PRU": st.column_config.NumberColumn("Prix Revient", format="%.2f €"),
                "Prix_Actuel": st.column_config.NumberColumn("Cours", format="%.2f €"),
                "Valo": st.column_config.NumberColumn("Valorisation", format="%.2f €"),
                "Perf_%": st.column_config.ProgressColumn(
                    "Performance", 
                    format="%.2f %%", 
                    min_value=-20, 
                    max_value=50,
                    help="Barre verte = PV, Rouge = MV"
                ),
            },
            hide_index=True,
            use_container_width=True
        )
        
    with col_graph:
        st.subheader("Répartition")
        # Graphique Plotly "Clean" (Template simple_white)
        fig_donut = px.donut(
            df[df['Ticker'] != "CASH"], 
            values='Valo', 
            names='Nom', 
            hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_layout(showlegend=False, template="simple_white", margin=dict(t=0, b=0, l=0, r=0))
        # Ajout du total au centre
        fig_donut.add_annotation(text=f"{total_valo/1000:.1f}k€", showarrow=False, font=dict(size=20, color="#0f172a"))
        st.plotly_chart(fig_donut, use_container_width=True)

# ONGLET 2 : ANALYSE (WATERFALL)
with tab_analytics:
    st.subheader("Contribution à la performance")
    
    fig_water = go.Figure(go.Waterfall(
        orientation = "v",
        measure = ["relative"] * len(df),
        x = df['Nom'],
        y = df['Plus_Value'],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#ef4444"}}, # Rouge mat
        increasing = {"marker":{"color":"#10b981"}}, # Vert émeraude
        totals = {"marker":{"color":"#3b82f6"}}
    ))
    fig_water.update_layout(
        title="Gains/Pertes par ligne (€)",
        template="simple_white",
        showlegend=False
    )
    st.plotly_chart(fig_water, use_container_width=True)

# ONGLET 3 : SIMULATEUR
with tab_simu:
    col_input, col_res = st.columns([1, 3])
    
    with col_input:
        st.markdown("#### ⚙️ Paramètres")
        apport = st.number_input("Épargne mensuelle (€)", 100, 5000, 500, step=100)
        taux = st.slider("Rendement annuel (%)", 1.0, 15.0, 8.0, 0.5)
        duree = st.slider("Horizon (Années)", 5, 40, 15)
    
    with col_res:
        years = range(datetime.now().year, datetime.now().year + duree + 1)
        vals = [total_valo]
        interests_acc = [0]
        
        for _ in range(duree):
            interest = vals[-1] * (taux/100)
            new_val = vals[-1] + interest + (apport * 12)
            vals.append(new_val)
            interests_acc.append(interests_acc[-1] + interest)
            
        df_proj = pd.DataFrame({"Année": years, "Capital Total": vals, "Dont Intérêts": interests_acc})
        
        st.line_chart(df_proj, x="Année", y=["Capital Total", "Dont Intérêts"], color=["#0f172a", "#10b981"])
        st.success(f"🎯 Capital final estimé : **{vals[-1]:,.0f} €** (dont {interests_acc[-1]:,.0f} € d'intérêts composés)")