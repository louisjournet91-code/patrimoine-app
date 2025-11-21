import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Patrimoine & Liberté", layout="wide", page_icon="🏛️")

# --- STYLE CSS PREMIUM ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center;}
    .stTabs [data-baseweb="tab-list"] {gap: 20px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 5px; font-weight: 600;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #e6f3ff; color: #0068c9;}
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS UTILITAIRES ---
def get_yahoo_price(ticker):
    """Récupère le dernier prix de clôture via Yahoo Finance."""
    try:
        if ticker == "CASH": return 1.0
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return 0.0
    except:
        return 0.0

# --- CHARGEMENT DES DONNÉES (SIMULATION BASÉE SUR VOS CSV) ---
# Dans une version avancée, nous lirons directement vos CSV uploadés.
# Ici, je reconstruis la structure de "PEA Marie" en dur pour le démarrage.

def load_portfolio_data():
    # Structure basée sur votre fichier "Epargne - PEA Marie.csv"
    data = {
        "Ticker": ["EPA:ESE", "EPA:CW8", "BTC-EUR", "EPA:PANX", "CASH"],
        "Nom": ["BNP S&P 500", "Amundi MSCI World", "Bitcoin", "Amundi Nasdaq", "Liquidités"],
        "Catégorie": ["ETF USA", "ETF Monde", "Crypto", "ETF Tech", "Cash"],
        "Quantité": [141, 10, 0.5, 20, 510.84], # Valeurs exemples
        "PRU": [24.41, 400.00, 30000.00, 600.00, 1.00], # Prix de Revient Unitaire
        "Objectif_%": [40, 30, 10, 15, 5] # Votre allocation cible (Config)
    }
    return pd.DataFrame(data)

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Gestion Privée")
    st.markdown("---")
    st.caption("Dernière mise à jour : " + datetime.now().strftime("%d/%m/%Y"))
    
    st.header("📁 Import de données")
    uploaded_file = st.file_uploader("Mettre à jour l'Historique (CSV)", type="csv")
    
    st.markdown("---")
    st.info("💡 Astuce : Les prix sont mis à jour en temps réel via Yahoo Finance.")

# --- TABS (LES ONGLETS) ---
tab1, tab2, tab3 = st.tabs(["📈 PEA & Actifs", "⏳ Historique & Évolution", "⚙️ Configuration & Projections"])

# ==============================================================================
# ONGLET 1 : PEA MARIE (Tableau de bord principal)
# ==============================================================================
with tab1:
    st.header("Vue détaillée du Portefeuille")
    
    # 1. Chargement et Calculs
    df = load_portfolio_data()
    
    # Conversion des tickers pour Yahoo (EPA: -> .PA)
    df['Yahoo_Ticker'] = df['Ticker'].apply(lambda x: x.replace("EPA:", "") + ".PA" if "EPA:" in x else x)
    
    # Récupération des prix en direct
    with st.spinner('Récupération des cours de bourse...'):
        df['Prix_Actuel'] = df['Yahoo_Ticker'].apply(get_yahoo_price)
    
    # Calculs financiers
    df['Valo_Totale'] = df['Quantité'] * df['Prix_Actuel']
    df['Investi_Total'] = df['Quantité'] * df['PRU']
    df['Plus_Value'] = df['Valo_Totale'] - df['Investi_Total']
    df['Performance_%'] = ((df['Prix_Actuel'] - df['PRU']) / df['PRU']) * 100
    
    # Totaux
    total_valo = df['Valo_Totale'].sum()
    total_investi = df['Investi_Total'].sum()
    plus_value_globale = total_valo - total_investi
    perf_globale = (plus_value_globale / total_investi) * 100
    
    # 2. KPI (Indicateurs Clés)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valorisation Totale", f"{total_valo:,.2f} €", delta=None)
    c2.metric("Montant Investi", f"{total_investi:,.2f} €")
    c3.metric("Plus-Value Latente", f"{plus_value_globale:,.2f} €", f"{perf_globale:.2f} %")
    c4.metric("Volatilité (VIX)", "20.55", "-1.5%") # À connecter plus tard
    
    st.markdown("---")
    
    # 3. Graphiques et Tableau
    col_gauche, col_droite = st.columns([2, 1])
    
    with col_gauche:
        st.subheader("Composition du Portefeuille")
        # Tableau stylisé
        st.dataframe(
            df[['Nom', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo_Totale', 'Performance_%']].style.format({
                'PRU': "{:.2f} €",
                'Prix_Actuel': "{:.2f} €",
                'Valo_Totale': "{:.2f} €",
                'Performance_%': "{:+.2f} %"
            }).background_gradient(subset=['Performance_%'], cmap='RdYlGn', vmin=-10, vmax=20),
            use_container_width=True,
            height=300
        )
        
    with col_droite:
        st.subheader("Allocation Sectorielle")
        fig_pie = px.donut(df, values='Valo_Totale', names='Catégorie', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==============================================================================
# ONGLET 2 : HISTORIQUE (Analyse temporelle)
# ==============================================================================
with tab2:
    st.header("Évolution du Patrimoine")
    
    # Si l'utilisateur a uploadé son fichier "Historique.csv"
    if uploaded_file is not None:
        try:
            hist_df = pd.read_csv(uploaded_file, sep=';') # Attention au séparateur CSV (souvent ; en France)
            # Convertir la date
            hist_df['Date'] = pd.to_datetime(hist_df['Date'], dayfirst=True)
            st.success("Fichier Historique chargé avec succès !")
            
            # Graphique d'évolution
            fig_hist = px.area(hist_df, x='Date', y='Montant', title="Croissance du Capital")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")
    else:
        st.info("Veuillez glisser votre fichier 'Epargne - Historique.csv' dans la barre latérale pour voir vos données.")
        
        # Données factices pour montrer l'exemple en attendant
        dates = pd.date_range(start="2023-01-01", periods=12, freq='M')
        vals = [15000, 15500, 16200, 15800, 16500, 17200, 17000, 17900, 18500, 19000, 19500, 17952]
        dummy_df = pd.DataFrame({'Date': dates, 'Valorisation': vals})
        
        fig_dummy = px.line(dummy_df, x='Date', y='Valorisation', markers=True)
        fig_dummy.update_traces(line_color='#0068c9', line_width=3)
        st.plotly_chart(fig_dummy, use_container_width=True)

# ==============================================================================
# ONGLET 3 : CONFIGURATION & RETRAITE
# ==============================================================================
with tab3:
    st.header("Simulateur de Liberté Financière")
    
    col_conf1, col_conf2 = st.columns(2)
    
    with col_conf1:
        st.subheader("Paramètres de Projection")
        apport_mensuel = st.slider("Apport Mensuel (€)", 0, 5000, 500, step=50)
        rendement_annuel = st.slider("Rendement Annuel Espéré (%)", 2.0, 15.0, 8.0, step=0.5)
        annees = st.slider("Horizon (Années)", 5, 30, 15)
        
    with col_conf2:
        st.subheader("Résultats")
        
        # Calcul des intérêts composés
        capital = total_valo
        data_proj = []
        
        for i in range(1, annees + 1):
            interets = capital * (rendement_annuel / 100)
            capital = capital + interets + (apport_mensuel * 12)
            data_proj.append({"Année": datetime.now().year + i, "Capital": capital, "Intérêts": interets})
            
        df_proj = pd.DataFrame(data_proj)
        
        final_capital = df_proj.iloc[-1]['Capital']
        st.metric(label=f"Capital en {df_proj.iloc[-1]['Année']}", value=f"{final_capital:,.0f} €")
        st.success(f"Avec un rendement de {rendement_annuel}%, vos intérêts seuls paieront {(final_capital*0.04)/12:,.0f} €/mois (Règle des 4%).")

    st.bar_chart(df_proj, x="Année", y="Capital", color="#2ecc71")