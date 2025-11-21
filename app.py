import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# --- 1. CONFIGURATION DU QUARTIER GÉNÉRAL ---
st.set_page_config(
    page_title="Cabinet Privé - Gestion de Patrimoine",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# --- 2. STYLE "HAUTE COUTURE" (CSS) ---
st.markdown("""
<style>
    /* Fond global et polices */
    .main { background-color: #0e1117; color: #fafafa; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Cartes de métriques (KPI) */
    div[data-testid="metric-container"] {
        background-color: #1e232d;
        border: 1px solid #303642;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover { transform: scale(1.02); border-color: #d4af37; }
    
    /* Titres dorés pour l'élégance */
    h1, h2, h3 { color: #ffffff; font-weight: 300; letter-spacing: 1px; }
    .gold-text { color: #d4af37; font-weight: bold; }
    
    /* Tableaux */
    .stDataFrame { border: 1px solid #303642; border-radius: 5px; }
    
    /* Onglets personnalisés */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1e232d;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #303642;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #d4af37;
        color: #000000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MOTEUR FINANCIER (LOGIQUE & DATA) ---

@st.cache_data(ttl=300)  # Cache de 5 minutes pour la rapidité
def get_live_prices(tickers_list):
    """Récupère les prix en lot pour optimiser la performance."""
    # Séparation Cash / Tickers
    valid_tickers = [t for t in tickers_list if t != "CASH"]
    
    if not valid_tickers:
        return {}
        
    try:
        # Téléchargement groupé
        data = yf.download(valid_tickers, period="1d", progress=False)['Close']
        prices = {}
        
        # Gestion du format de retour de yfinance (Series vs DataFrame)
        if len(valid_tickers) == 1:
             prices[valid_tickers[0]] = data.iloc[-1]
        else:
            for t in valid_tickers:
                prices[t] = data[t].iloc[-1]
        
        prices["CASH"] = 1.0
        return prices
    except Exception as e:
        st.error(f"Erreur de connexion aux marchés : {e}")
        return {t: 0.0 for t in tickers_list}

def load_assets():
    """Structure du portefeuille (Votre État-Major)."""
    data = {
        "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
        "Nom": [
            "BNP S&P 500", 
            "Amundi ETF Div", 
            "Amundi USA x2", 
            "Amundi World x2", 
            "Bitcoin (Reserve)", 
            "Liquidités PEA"
        ],
        "Type": ["Action (ETF)", "Action (ETF)", "Action (Levier)", "Action (Levier)", "Crypto", "Cash"],
        "Risque": ["Moyen", "Moyen", "Élevé", "Élevé", "Spéculatif", "Nul"],
        "Quantité": [141, 71, 55, 176, 0.015, 510.84], # Légère augmentation BTC pour l'exemple
        "PRU": [24.41, 4.68, 71.73, 19.71, 65000.00, 1.00]
    }
    return pd.DataFrame(data)

def calculate_portfolio(df):
    prices = get_live_prices(df['Ticker'].tolist())
    df['Prix_Actuel'] = df['Ticker'].map(prices)
    df['Valo_Actuelle'] = df['Quantité'] * df['Prix_Actuel']
    df['Investissement'] = df['Quantité'] * df['PRU']
    df['Plus_Value'] = df['Valo_Actuelle'] - df['Investissement']
    df['Perf_%'] = ((df['Prix_Actuel'] - df['PRU']) / df['PRU']) * 100
    return df

# --- 4. BARRE LATÉRALE : CENTRE DE CONTRÔLE ---
with st.sidebar:
    st.markdown("### 🏛️ CABINET PRIVÉ")
    st.caption(f"Bienvenue, Monsieur.")
    st.markdown("---")
    
    # Indicateurs de marché en temps réel
    st.markdown("#### 🌍 Tendance Marchés")
    market_data = get_live_prices(["^FCHI", "^GSPC", "BTC-EUR"]) # CAC40, S&P500, BTC
    
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("CAC 40", f"{market_data.get('^FCHI', 0):.0f}", delta=None)
    col_s2.metric("S&P 500", f"{market_data.get('^GSPC', 0):.0f}", delta=None)
    st.metric("Bitcoin", f"{market_data.get('BTC-EUR', 0):,.0f} €", delta_color="normal")
    
    st.markdown("---")
    st.info("💡 **Note de synthèse** : L'inflation en France est stable. Pensez à renforcer vos positions sur le repli du levier USA.")

# --- 5. CORPS PRINCIPAL ---

st.title("Tableau de Bord Patrimonial")
st.markdown(f"*Dernière valorisation : {datetime.now().strftime('%d %B %Y à %H:%M')}*")

# Chargement des données
df_raw = load_assets()
df = calculate_portfolio(df_raw)

# KPI GLOBAUX (Ligne du haut)
total_assets = df['Valo_Actuelle'].sum()
total_invested = df['Investissement'].sum()
total_pnl = total_assets - total_invested
global_perf = (total_pnl / total_invested) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("VALORISATION NETTE", f"{total_assets:,.2f} €", "Net Worth")
c2.metric("CAPITAL INVESTI", f"{total_invested:,.2f} €", "Cash deployed")
c3.metric("PLUS-VALUE LATENTE", f"{total_pnl:+,.2f} €", f"{global_perf:+.2f} %")
c4.metric("OBJECTIF RETRAITE", f"{((total_assets/1000000)*100):.1f} %", "du 1er Million")

st.markdown("---")

# TABS
tab_pf, tab_alloc, tab_sim = st.tabs(["💼 PORTEFEUILLE DÉTAILLÉ", "📊 ANALYSE & RISQUE", "🚀 PROJECTION RENTE"])

# --- ONGLET 1 : LE PORTEFEUILLE ---
with tab_pf:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Inventaire des Actifs")
        # Configuration du tableau pour qu'il soit très lisible
        st.dataframe(
            df[['Nom', 'Type', 'PRU', 'Prix_Actuel', 'Valo_Actuelle', 'Plus_Value', 'Perf_%']].style
            .format({
                'PRU': "{:.2f} €",
                'Prix_Actuel': "{:.2f} €",
                'Valo_Actuelle': "{:,.2f} €",
                'Plus_Value': "{:+,.2f} €",
                'Perf_%': "{:+.2f} %"
            })
            .background_gradient(subset=['Perf_%'], cmap='RdYlGn', vmin=-10, vmax=30)
            .bar(subset=['Valo_Actuelle'], color='#d4af37'),
            use_container_width=True,
            height=400
        )

    with col_right:
        st.subheader("Répartition par Poids")
        # Treemap : plus professionnel que le "camembert" pour voir les masses
        fig_tree = px.treemap(
            df, 
            path=['Type', 'Nom'], 
            values='Valo_Actuelle',
            color='Perf_%',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_tree, use_container_width=True)

# --- ONGLET 2 : ANALYSE ---
with tab_alloc:
    st.subheader("Exposition Sectorielle et Risque")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        # Waterfall Chart pour visualiser la construction de la valeur
        fig_water = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative"] * len(df),
            x = df['Nom'],
            textposition = "outside",
            text = [f"{v/1000:.1f}k" for v in df['Valo_Actuelle']],
            y = df['Valo_Actuelle'],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_water.update_layout(
            title = "Poids relatif des actifs (Waterfall)",
            showlegend = False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig_water, use_container_width=True)
        
    with col_a2:
        # Graphique Radar pour le risque (Simulation)
        categories = ['Diversification', 'Volatilité', 'Liquidité', 'Rendement', 'Protection Inflation']
        # Note subjective basée sur votre profil
        values = [3, 4, 5, 4, 3] 
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line_color='#d4af37'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="Radar de Solidité du Portefeuille",
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# --- ONGLET 3 : LIBERTÉ FINANCIÈRE ---
with tab_sim:
    st.subheader("Planification de la Rente")
    st.markdown("Projection basée sur l'accumulation de capital et la règle des 4% (Trinity Study).")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        with st.container(border=True):
            st.markdown("#### ⚙️ Paramètres")
            apport_mensuel = st.number_input("Épargne Mensuelle (€)", value=1000, step=100)
            rendement = st.slider("Rendement Annuel Moyen (%)", 2.0, 12.0, 8.0)
            annees = st.slider("Horizon (Années)", 5, 30, 15)
            inflation = st.slider("Inflation estimée (%)", 0.0, 5.0, 2.0)
            
            st.markdown("---")
            st.caption("Le rendement réel sera calculé net d'inflation.")

    with col_sim2:
        # Calculs vectorisés
        rendement_reel = (1 + rendement/100) / (1 + inflation/100) - 1
        dates = range(datetime.now().year, datetime.now().year + annees + 1)
        capital_proj = [total_assets]
        
        for _ in range(annees):
            # Capital N-1 * Rendement + Apport Annuel
            new_cap = capital_proj[-1] * (1 + rendement_reel) + (apport_mensuel * 12)
            capital_proj.append(new_cap)
            
        df_proj = pd.DataFrame({"Année": dates, "Capital": capital_proj})
        rente_mensuelle = (capital_proj[-1] * 0.04) / 12
        
        # Graphique de projection
        fig_line = px.area(df_proj, x="Année", y="Capital", title="Évolution de la Fortune Nette (Ajustée Inflation)")
        fig_line.update_traces(line_color='#d4af37', fill_color='rgba(212, 175, 55, 0.2)')
        fig_line.add_hline(y=1000000, line_dash="dot", line_color="white", annotation_text="Target 1M€")
        fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Verdict
        st.success(f"🎯 **Résultat :** Dans {annees} ans, votre patrimoine estimé sera de **{capital_proj[-1]:,.0f} €** (pouvoir d'achat d'aujourd'hui).")
        st.markdown(f"Cela générerait une **rente passive 'infinie' de {rente_mensuelle:,.0f} € / mois** sans toucher au capital.")