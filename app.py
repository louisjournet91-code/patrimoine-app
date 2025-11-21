import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURATION DU QUARTIER GÉNÉRAL ---
st.set_page_config(
    page_title="Cabinet Privé - Gestion de Patrimoine",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# Astuce : On force le thème sombre pour que votre CSS fonctionne partout
# (Normalement cela se fait dans un fichier .toml, mais ceci aide visuellement)
st.markdown("""
<style>
    /* Force le texte blanc par défaut pour contrer le mode clair */
    body { color: #fafafa; }
    
    /* Fond global et polices */
    .stApp { background-color: #0e1117; }
    
    /* Cartes de métriques (KPI) */
    div[data-testid="stMetric"] {
        background-color: #1e232d;
        border: 1px solid #303642;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"]:hover { border-color: #d4af37; }
    
    /* Titres dorés */
    h1, h2, h3, h4 { color: #ffffff !important; font-weight: 300; }
    .gold-text { color: #d4af37; font-weight: bold; }
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
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

# --- 2. MOTEUR DE DONNÉES ROBUSTE ---

@st.cache_data(ttl=600) # Cache augmenté à 10min pour performance
def get_live_prices(tickers_list):
    """Récupération blindée des prix Yahoo Finance."""
    valid_tickers = [t for t in tickers_list if t != "CASH"]
    prices = {"CASH": 1.0}
    
    if not valid_tickers:
        return prices
        
    try:
        # Téléchargement groupé
        data = yf.download(valid_tickers, period="1d", progress=False)['Close']
        
        # CAS 1 : Un seul ticker (renvoie une Series)
        if isinstance(data, pd.Series):
            val = data.iloc[-1]
            prices[valid_tickers[0]] = float(val)
            
        # CAS 2 : Plusieurs tickers (renvoie un DataFrame)
        elif isinstance(data, pd.DataFrame):
            # On s'assure de prendre la dernière ligne valide
            last_row = data.iloc[-1]
            for ticker in valid_tickers:
                if ticker in last_row.index:
                    prices[ticker] = float(last_row[ticker])
                else:
                    prices[ticker] = 0.0 # Sécurité si ticker non trouvé
                    
    except Exception as e:
        st.error(f"Erreur API Yahoo : {e}")
    
    return prices

def get_portfolio_data():
    """Charge les données : Soit depuis le CSV uploadé, soit les données démo."""
    if 'portfolio_csv' in st.session_state and st.session_state['portfolio_csv'] is not None:
        try:
            # Lecture du CSV utilisateur (attend ; ou , comme séparateur)
            df = pd.read_csv(st.session_state['portfolio_csv'], sep=None, engine='python')
            return df
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")
    
    # DONNÉES PAR DÉFAUT (VOTRE EXEMPLE)
    data = {
        "Ticker": ["EPA:ESE", "EPA:DCAM", "EPA:PUST", "EPA:CL2", "BTC-EUR", "CASH"],
        "Nom": ["BNP S&P 500", "Amundi ETF Div", "Amundi USA x2", "Amundi World x2", "Bitcoin", "Liquidités"],
        "Type": ["ETF Action", "ETF Action", "ETF Levier", "ETF Levier", "Crypto", "Cash"],
        "Quantité": [141, 71, 55, 176, 0.015, 510.84],
        "PRU": [24.41, 4.68, 71.73, 19.71, 65000.00, 1.00]
    }
    return pd.DataFrame(data)

# --- 3. CALCULS FINANCIERS ---
def calculate_portfolio(df):
    # Nettoyage des tickers pour Yahoo (EPA: -> .PA)
    df['Yahoo_Ticker'] = df['Ticker'].apply(lambda x: x.replace("EPA:", "") + ".PA" if "EPA:" in x else x)
    
    # Récupération prix
    prices = get_live_prices(df['Yahoo_Ticker'].unique().tolist())
    
    df['Prix_Actuel'] = df['Yahoo_Ticker'].apply(lambda x: prices.get(x, 0.0))
    
    # Calculs
    df['Valo_Actuelle'] = df['Quantité'] * df['Prix_Actuel']
    df['Investissement'] = df['Quantité'] * df['PRU']
    df['Plus_Value'] = df['Valo_Actuelle'] - df['Investissement']
    
    # Sécurité division par zéro
    df['Perf_%'] = df.apply(lambda row: ((row['Prix_Actuel'] - row['PRU']) / row['PRU']) * 100 if row['PRU'] > 0 else 0, axis=1)
    
    return df

# --- 4. INTERFACE : SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910282.png", width=60) # Petite icône dorée
    st.title("CABINET PRIVÉ")
    st.caption(f"Bienvenue, Monsieur.")
    st.markdown("---")
    
    # UPLOADER CSV (La clé de la liberté)
    st.markdown("#### 📂 Vos Actifs")
    uploaded_file = st.file_uploader("Mettre à jour (CSV)", type=['csv'], key='portfolio_csv')
    
    st.markdown("---")
    
    # Indicateurs Marché (Ticker corrigés pour Yahoo)
    st.markdown("#### 🌍 Indices")
    market_data = get_live_prices(["^FCHI", "^GSPC", "BTC-EUR"])
    
    c1, c2 = st.columns(2)
    c1.metric("CAC 40", f"{market_data.get('^FCHI',0):.0f}")
    c2.metric("S&P 500", f"{market_data.get('^GSPC',0):.0f}")
    st.metric("Bitcoin", f"{market_data.get('BTC-EUR',0):,.0f} €")

# --- 5. INTERFACE : MAIN ---
st.title("Tableau de Bord Patrimonial")
st.markdown(f"*Valorisation au {datetime.now().strftime('%d %B %Y')}*")

# Traitement
df_raw = get_portfolio_data()
df = calculate_portfolio(df_raw)

# KPI GLOBAUX
total_assets = df['Valo_Actuelle'].sum()
total_invested = df['Investissement'].sum()
total_pnl = total_assets - total_invested
perf_global = (total_pnl / total_invested) * 100 if total_invested > 0 else 0

# Affichage KPI
col1, col2, col3, col4 = st.columns(4)
col1.metric("PATRIMOINE NET", f"{total_assets:,.2f} €")
col2.metric("TOTAL INVESTI", f"{total_invested:,.2f} €")
col3.metric("PLUS-VALUE", f"{total_pnl:+,.2f} €", f"{perf_global:+.2f} %")
col4.metric("LIQUIDITÉS", f"{df[df['Ticker']=='CASH']['Valo_Actuelle'].sum():,.2f} €")

st.markdown("---")

# ONGLETS
tab1, tab2, tab3 = st.tabs(["💼 PORTEFEUILLE", "📊 ANALYSE", "🚀 PROJECTION"])

with tab1:
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.subheader("Positions Actuelles")
        # Tableau stylisé avec Pandas
        display_cols = ['Nom', 'Type', 'PRU', 'Prix_Actuel', 'Valo_Actuelle', 'Perf_%']
        
        st.dataframe(
            df[display_cols].style
            .format({
                'PRU': "{:.2f} €", 'Prix_Actuel': "{:.2f} €", 
                'Valo_Actuelle': "{:,.2f} €", 'Perf_%': "{:+.2f} %"
            })
            .background_gradient(subset=['Perf_%'], cmap='RdYlGn', vmin=-10, vmax=20),
            use_container_width=True, height=400
        )
        
    with c_right:
        st.subheader("Allocation")
        # On exclut le cash du donut pour mieux voir les risques
        df_invest = df[df['Ticker'] != 'CASH']
        fig = px.donut(df_invest, values='Valo_Actuelle', names='Nom', hole=0.5, color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Analyse de Performance")
    # Waterfall Chart (Très pro pour voir d'où vient la richesse)
    fig_water = go.Figure(go.Waterfall(
        orientation = "v",
        measure = ["relative"] * len(df),
        x = df['Nom'],
        textposition = "outside",
        text = [f"{x/1000:.1f}k" for x in df['Plus_Value']],
        y = df['Plus_Value'],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig_water.update_layout(title="Contribution à la Plus-Value (€)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_water, use_container_width=True)

with tab3:
    st.subheader("Simulateur de Rente")
    col_sim1, col_sim2 = st.columns([1,2])
    
    with col_sim1:
        apport = st.number_input("Épargne mensuelle (€)", value=1000, step=100)
        taux = st.slider("Taux annuel (%)", 2.0, 12.0, 8.0)
        annees = st.slider("Durée (ans)", 5, 30, 15)
    
    with col_sim2:
        dates = range(datetime.now().year, datetime.now().year + annees + 1)
        capital = [total_assets]
        for _ in range(annees):
            capital.append(capital[-1] * (1 + taux/100) + (apport * 12))
            
        df_proj = pd.DataFrame({"Année": dates, "Capital": capital})
        st.area_chart(df_proj.set_index("Année"), color="#d4af37")
        
        final_cap = capital[-1]
        st.success(f"Capital final estimé : **{final_cap:,.0f} €**. Cela génère une rente passive de **{(final_cap*0.04)/12:,.0f} €/mois**.")