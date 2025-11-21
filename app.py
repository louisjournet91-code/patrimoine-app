import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURATION & STYLE "BANQUE PRIVÉE" ---
st.set_page_config(
    page_title="Gestion Patrimoniale",
    layout="wide",
    page_icon="🏛️"
)

# CSS : Design Épuré (Fond clair, Lisibilité maximale)
st.markdown("""
<style>
    /* Fond global */
    .stApp { background-color: #f8f9fa; color: #212529; }
    
    /* Cartes (Containers) */
    div[data-testid="stMetric"], div.stDataFrame, div.stPlotlyChart {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Titres */
    h1, h2, h3 { color: #0f172a; font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 8px;
        color: #64748b;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0f172a;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LES DONNÉES OFFICIELLES (CORRIGÉES) ---
def load_portfolio():
    """Données certifiées conformes à votre dernier envoi."""
    data = {
        "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
        "Nom": [
            "BNP S&P 500 (ESE)", 
            "Amundi MSCI World (DCAM)", 
            "Lyxor NASDAQ-100 (PUST)", 
            "Amundi USA Levier x2 (CL2)", 
            "Bitcoin (Réserve)", 
            "Liquidités Espèces"
        ],
        "Type": ["ETF Action", "ETF Action", "ETF Tech", "ETF Levier", "Crypto", "Cash"],
        
        # Quantités exactes recalculées selon vos valorisations
        "Quantité": [
            141,      # ESE
            716,      # DCAM (3759€ / 5.25€ = 716 parts)
            55,       # PUST
            176,      # CL2
            0.01,     # BTC (Quantité ajustée selon votre tableau)
            510.84    # Cash (Reliquat précédent, modifiable)
        ],
        
        # Vos Prix de Revient Unitaires (PRU) Officiels
        "PRU": [
            24.41,      # ESE
            4.68,       # DCAM
            71.73,      # PUST
            19.71,      # CL2
            90165.46,   # BTC (PRU élevé noté)
            1.00
        ]
    }
    return pd.DataFrame(data)

@st.cache_data(ttl=600)
def get_prices(df):
    """Récupération des cours actuels."""
    tickers = [t for t in df['Ticker'].unique() if t != "CASH"]
    prices = {"CASH": 1.0}
    
    if tickers:
        try:
            # Téléchargement silencieux
            data = yf.download(tickers, period="1d", progress=False)['Close']
            
            # Gestion robuste du format de retour yfinance
            if hasattr(data, 'columns'): # Si plusieurs tickers
                last_row = data.iloc[-1]
                for t in tickers:
                    # On gère le cas où un ticker échoue
                    prices[t] = float(last_row[t]) if t in last_row else 0.0
            else: # Si un seul ticker
                prices[tickers[0]] = float(data.iloc[-1])
        except Exception as e:
            pass 
            
    return prices

# --- 3. CALCULS ---
df = load_portfolio()
market_prices = get_prices(df)

# Application des prix (Priorité au marché, sinon PRU si erreur réseau)
df['Prix_Actuel'] = df['Ticker'].apply(lambda x: market_prices.get(x, df.loc[df['Ticker']==x, 'PRU'].values[0]))

# Calculs de performance
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['Plus_Value'] = df['Valo'] - df['Investi']
# Gestion de la division par zéro pour la performance
df['Perf_%'] = df.apply(lambda x: ((x['Prix_Actuel'] - x['PRU']) / x['PRU'] * 100) if x['PRU'] > 0 else 0, axis=1)

# Totaux Généraux
total_valo = df['Valo'].sum()
total_investi = df['Investi'].sum()
total_pv = total_valo - total_investi
total_perf = (total_pv / total_investi) * 100

# --- 4. AFFICHAGE TABLEAU DE BORD ---

c1, c2 = st.columns([3,1])
with c1:
    st.title("Synthèse Patrimoniale")
    st.caption(f"Situation arrêtée au {datetime.now().strftime('%d/%m/%Y à %H:%M')}")

# KPI
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Patrimoine Net", f"{total_valo:,.2f} €")
kpi2.metric("Capital Investi", f"{total_investi:,.2f} €")
kpi3.metric("Plus-Value Latente", f"{total_pv:+,.2f} €", f"{total_perf:+.2f} %")
kpi4.metric("Liquidités", f"{df[df['Ticker']=='CASH']['Valo'].sum():,.2f} €")

st.markdown("---")

# ONGLETS
tab_pf, tab_graph, tab_proj = st.tabs(["📋 Détail Positions", "📊 Analyse", "🔮 Projection"])

# ONGLET 1 : LE TABLEAU DÉTAILLÉ
with tab_pf:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Inventaire")
        # Configuration avancée du tableau
        st.dataframe(
            df[['Nom', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo', 'Perf_%']],
            column_config={
                "Nom": st.column_config.TextColumn("Actif", width="medium"),
                "PRU": st.column_config.NumberColumn("Prix Achat", format="%.2f €"),
                "Prix_Actuel": st.column_config.NumberColumn("Cours Actuel", format="%.2f €"),
                "Valo": st.column_config.NumberColumn("Valorisation", format="%.2f €"),
                "Perf_%": st.column_config.ProgressColumn(
                    "Performance", 
                    format="%.2f %%", 
                    min_value=-30, max_value=30,
                    help="Vert = Gain, Rouge = Perte"
                ),
            },
            hide_index=True,
            use_container_width=True
        )
    
    with col_right:
        st.subheader("Poids dans le portefeuille")
        # Graphique Donut
        df_invest = df[df['Ticker'] != "CASH"]
        fig = px.donut(df_invest, values='Valo', names='Nom', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        fig.add_annotation(text=f"{total_valo/1000:.1f}k€", showarrow=False, font=dict(size=20))
        st.plotly_chart(fig, use_container_width=True)

# ONGLET 2 : ANALYSE DE PERFORMANCE (WATERFALL)
with tab_graph:
    st.subheader("Contribution aux Gains/Pertes")
    
    fig_water = go.Figure(go.Waterfall(
        orientation = "v",
        measure = ["relative"] * len(df),
        x = df['Nom'],
        y = df['Plus_Value'],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#ef4444"}}, # Rouge pour les pertes (BTC)
        increasing = {"marker":{"color":"#10b981"}}, # Vert pour les gains
    ))
    fig_water.update_layout(title="Plus-Values par Ligne (€)", template="simple_white")
    st.plotly_chart(fig_water, use_container_width=True)

# ONGLET 3 : PROJECTION
with tab_proj:
    c_sim1, c_sim2 = st.columns([1, 3])
    with c_sim1:
        st.markdown("#### Paramètres")
        apport = st.number_input("Apport mensuel (€)", 100, 5000, 500, step=100)
        taux = st.slider("Rendement (%)", 2.0, 15.0, 8.0)
        annees = st.slider("Horizon (ans)", 5, 30, 15)
        
    with c_sim2:
        vals = [total_valo]
        years = range(datetime.now().year, datetime.now().year + annees + 1)
        
        for _ in range(annees):
            vals.append(vals[-1] * (1 + taux/100) + (apport * 12))
            
        df_proj = pd.DataFrame({"Année": years, "Capital": vals})
        st.line_chart(df_proj, x="Année", y="Capital", color="#0f172a")
        st.success(f"🎯 Capital final estimé : **{vals[-1]:,.0f} €**")