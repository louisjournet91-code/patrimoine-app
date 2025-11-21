import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="Gestion Patrimoniale", layout="wide", page_icon="🏛️")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    div[data-testid="stMetric"], div.stDataFrame, div.stPlotlyChart {
        background-color: #ffffff; border: 1px solid #e9ecef; padding: 15px;
        border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #0f172a; font-family: 'Helvetica Neue', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #ffffff; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #0f172a; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. DONNÉES & PARAMÈTRES ---

# PARAMÈTRES FIXES
MONTANT_INITIAL_CASH = 15450.00
DATE_DEBUT = datetime(2022, 1, 1)

def load_portfolio():
    data = {
        "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
        "Nom": ["BNP S&P 500", "Amundi World", "Lyxor Nasdaq", "Amundi USA x2", "Bitcoin", "Liquidités"],
        "Type": ["ETF Action", "ETF Action", "ETF Tech", "ETF Levier", "Crypto", "Cash"],
        "Quantité": [141, 716, 55, 176, 0.01, 510.84],
        "PRU": [24.41, 4.68, 71.73, 19.71, 90165.46, 1.00]
    }
    return pd.DataFrame(data)

@st.cache_data(ttl=300)
def get_market_data(df):
    """Récupère Prix Actuel ET Prix Veille"""
    tickers = [t for t in df['Ticker'].unique() if t != "CASH"]
    current_prices = {"CASH": 1.0}
    prev_prices = {"CASH": 1.0}
    
    if tickers:
        try:
            # Téléchargement groupé
            hist = yf.download(tickers, period="5d", progress=False)['Close']
            
            # Gestion format retour yfinance
            if hasattr(hist, 'columns') and len(hist.columns) > 1:
                last_row = hist.iloc[-1]
                prev_row = hist.iloc[-2]
                for t in tickers:
                    if t in last_row:
                        current_prices[t] = float(last_row[t])
                        prev_prices[t] = float(prev_row[t])
            else: # Cas un seul ticker ou structure différente
                # Fallback simple si structure complexe
                pass 
                
        except Exception as e:
            pass
            
    return current_prices, prev_prices

# --- 3. CALCULS ---
df = load_portfolio()
cur_prices, prev_prices = get_market_data(df)

# Injection des prix (Sécurité : Si pas de prix réseau, on prend le PRU pour éviter 0)
df['Prix_Actuel'] = df['Ticker'].apply(lambda x: cur_prices.get(x, df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Prix_Veille'] = df['Ticker'].apply(lambda x: prev_prices.get(x, df.loc[df['Ticker']==x, 'PRU'].values[0]))

# Calculs
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['Plus_Value'] = df['Valo'] - df['Investi']
df['Perf_%'] = ((df['Prix_Actuel'] - df['PRU']) / df['PRU']) * 100
df['Valo_Veille'] = df['Quantité'] * df['Prix_Veille']
df['Var_Jour_€'] = df['Valo'] - df['Valo_Veille']

# Agrégats
cash_dispo = df[df['Ticker']=="CASH"]['Valo'].sum()
df_invested = df[df['Ticker']!="CASH"]
valo_investi = df_invested['Valo'].sum()
montant_investi_titres = df_invested['Investi'].sum()
portefeuille_total = valo_investi + cash_dispo

# Performances Globales
perf_totale_eur = portefeuille_total - MONTANT_INITIAL_CASH
perf_totale_pct = (perf_totale_eur / MONTANT_INITIAL_CASH) * 100
perf_actif_eur = valo_investi - montant_investi_titres
perf_actif_pct = (perf_actif_eur / montant_investi_titres) * 100 if montant_investi_titres > 0 else 0
pv_jour_eur = df['Var_Jour_€'].sum()
# Gestion division par zéro pour variation jour
base_jour = portefeuille_total - pv_jour_eur
pv_jour_pct = (pv_jour_eur / base_jour) * 100 if base_jour > 0 else 0

# CAGR
days_held = (datetime.now() - DATE_DEBUT).days
years_held = days_held / 365.25
cagr = ((portefeuille_total / MONTANT_INITIAL_CASH) ** (1/years_held) - 1) * 100 if years_held > 0 else 0

# --- 4. INTERFACE ---
c_header, c_date = st.columns([3,1])
with c_header:
    st.title("Synthèse de Gestion")
with c_date:
    st.markdown(f"<div style='text-align:right; padding-top:20px'><b>{datetime.now().strftime('%d/%m/%Y')}</b></div>", unsafe_allow_html=True)

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Portefeuille Total", f"{portefeuille_total:,.2f} €")
k2.metric("Montant Initial", f"{MONTANT_INITIAL_CASH:,.2f} €")
k3.metric("Liquidités", f"{cash_dispo:,.2f} €")
k4.metric("Valorisation Investi", f"{valo_investi:,.2f} €")

st.markdown("---")

p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Perf. Totale", f"{perf_totale_eur:+,.2f} €", f"{perf_totale_pct:+.2f}%")
p2.metric("Perf. Actifs", f"{perf_actif_eur:+,.2f} €", f"{perf_actif_pct:+.2f}%")
p3.metric("Var. Jour", f"{pv_jour_eur:+,.2f} €", f"{pv_jour_pct:+.2f}%")
p4.metric("Rendement/An", f"{perf_totale_pct/years_held:.2f}%")
p5.metric("CAGR", f"{cagr:.2f}%")

st.markdown("---")

# ONGLETS
tab_pf, tab_visu, tab_sim = st.tabs(["📋 Positions", "📊 Analyse Graphique", "🔮 Projection"])

with tab_pf:
    st.dataframe(
        df[['Nom', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo', 'Var_Jour_€', 'Perf_%']],
        column_config={
            "Nom": st.column_config.TextColumn("Actif"),
            "PRU": st.column_config.NumberColumn("PRU", format="%.2f €"),
            "Prix_Actuel": st.column_config.NumberColumn("Cours", format="%.2f €"),
            "Valo": st.column_config.NumberColumn("Valo", format="%.2f €"),
            "Var_Jour_€": st.column_config.NumberColumn("Var. Jour", format="%+.2f €"),
            "Perf_%": st.column_config.ProgressColumn("Perf.", format="%+.2f %%", min_value=-20, max_value=40)
        },
        hide_index=True, use_container_width=True
    )

with tab_visu:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        # Waterfall (CORRIGÉ: px.bar simple ou go.Waterfall)
        fig_w = go.Figure(go.Waterfall(
            orientation="v", measure=["relative"] * len(df),
            x=df['Nom'], y=df['Plus_Value'],
            connector={"line":{"color":"#cbd5e1"}},
            decreasing={"marker":{"color":"#ef4444"}}, increasing={"marker":{"color":"#10b981"}}
        ))
        fig_w.update_layout(title="Contribution PV (€)", template="simple_white")
        st.plotly_chart(fig_w, use_container_width=True)
    
    with col_g2:
        # Donut (CORRIGÉ: px.pie avec hole)
        fig_d = px.pie(
            df[df['Ticker']!="CASH"], 
            values='Valo', 
            names='Nom', 
            hole=0.6, 
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_d.update_layout(title="Répartition", showlegend=False, template="simple_white")
        fig_d.add_annotation(text=f"{valo_investi/1000:.1f}k€", showarrow=False, font=dict(size=20))
        st.plotly_chart(fig_d, use_container_width=True)

with tab_sim:
    vals = [portefeuille_total]
    annees = 15
    rendement_hyp = 8.0
    apport = 500
    for _ in range(annees):
        vals.append(vals[-1] * (1 + rendement_hyp/100) + (apport * 12))
    st.line_chart(vals)
    st.caption(f"Projection simple : +{apport}€/mois à {rendement_hyp}%/an")