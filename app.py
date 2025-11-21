import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import io

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tableau de Bord", layout="wide", page_icon="💎")

# --- 2. GESTION DU THÈME (DARK / LIGHT) ---
with st.sidebar:
    st.header("⚙️ Préférences")
    dark_mode = st.toggle("🌙 Mode Sombre", value=True) # Par défaut sombre pour l'effet "Premium"
    st.caption("Tableau de Bord V.1.0")

# Définition des palettes selon le mode
if dark_mode:
    # --- THEME MIDNIGHT ONYX (SOMBRE) ---
    bg_color = "#0f172a"
    text_color = "#f8fafc"
    card_bg = "rgba(30, 41, 59, 0.7)"
    border_color = "rgba(255, 255, 255, 0.1)"
    chart_line_color = "#38bdf8" # Cyan électrique
    chart_fill_color = "rgba(56, 189, 248, 0.15)"
    metric_gradient = "linear-gradient(135deg, #38bdf8 0%, #818cf8 100%)"
    
    css_theme = """
    /* FOND SOMBRE PROFOND */
    .stApp {
        background-color: #020617;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        color: #f8fafc;
    }
    h1, h2, h3, p, span, div { color: #f8fafc; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    """
else:
    # --- THEME LIQUID DAYLIGHT (CLAIR) ---
    bg_color = "#f0f4f8"
    text_color = "#1e293b"
    card_bg = "rgba(255, 255, 255, 0.6)"
    border_color = "rgba(255, 255, 255, 0.8)"
    chart_line_color = "#2563eb" # Bleu roi
    chart_fill_color = "rgba(37, 99, 235, 0.1)"
    metric_gradient = "linear-gradient(135deg, #0f172a 0%, #334155 100%)"
    
    css_theme = """
    /* FOND MAILLÉ CLAIR */
    .stApp {
        background-color: #f0f4f8;
        background-image: 
            radial-gradient(at 0% 0%, hsla(210,100%,96%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(220,100%,93%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(190,100%,94%,1) 0, transparent 50%);
        color: #1e293b;
    }
    h1, h2, h3 { color: #0f172a; }
    div[data-testid="stMetricLabel"] { color: #64748b !important; }
    """

# Injection du CSS Dynamique
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');
    
    .stApp {{ font-family: 'Outfit', sans-serif; }}

    {css_theme}

    /* --- STYLE GLASS UNIFIÉ --- */
    div[data-testid="stMetric"], 
    div[data-testid="stDataFrame"], 
    div.stPlotlyChart, 
    div.stExpander {{
        background: {card_bg} !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 24px;
        border: 1px solid {border_color};
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06);
        padding: 24px !important;
        transition: transform 0.3s ease;
    }}

    div[data-testid="stMetric"]:hover {{ transform: translateY(-5px); }}

    /* TYPO & VALEURS */
    h1, h2, h3 {{ font-weight: 800; letter-spacing: -0.5px; }}
    
    div[data-testid="stMetricLabel"] {{
        font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 32px; font-weight: 800;
        background: {metric_gradient};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    
    /* COMPOSANTS UI */
    .section-header {{
        margin-top: 40px; margin-bottom: 20px; font-size: 24px; font-weight: 700;
        border-bottom: 2px solid {border_color}; padding-bottom: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# --- 3. FONCTIONS ROBUSTES (READ-ONLY) ---

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# Portefeuille par défaut (Sécurité démarrage)
INITIAL_PORTFOLIO = {
    "Ticker": ["ESE.PA", "CASH"],
    "Nom": ["S&P 500", "Liquidités"],
    "Type": ["ETF", "Cash"],
    "Quantité": [10.0, 1000.0],
    "PRU": [25.0, 1.0]
}

def safe_float(x):
    """Conversion blindée"""
    if pd.isna(x) or x == "": return 0.0
    s = str(x).strip().replace('"', '').replace('%', '').replace('€', '').replace(' ', '')
    try: return float(s.replace(',', '.'))
    except: return 0.0

def load_data():
    """Chargement unique des données (Read-Only)"""
    # 1. Portefeuille
    if os.path.exists(FILE_PORTFOLIO):
        try:
            df = pd.read_csv(FILE_PORTFOLIO, sep=';', dtype=str)
            if df.shape[1] < 2: df = pd.read_csv(FILE_PORTFOLIO, sep=',', dtype=str)
        except: df = pd.DataFrame(INITIAL_PORTFOLIO)
    else:
        df = pd.DataFrame(INITIAL_PORTFOLIO)

    # Nettoyage types
    for c in ['Quantité', 'PRU']:
        if c in df.columns: df[c] = df[c].apply(safe_float)

    # 2. Historique
    hist_cols = ["Date", "Total", "Delta", "PF_Index100", "ESE_Index100"]
    if os.path.exists(FILE_HISTORY):
        try:
            df_h = pd.read_csv(FILE_HISTORY, sep=';', on_bad_lines='skip', engine='python')
            if df_h.shape[1] < 3: df_h = pd.read_csv(FILE_HISTORY, sep=',', on_bad_lines='skip')
            
            # Conversion numérique des colonnes critiques
            for c in df_h.columns:
                if c != "Date": df_h[c] = df_h[c].apply(safe_float)
            
            df_h['Date'] = pd.to_datetime(df_h['Date'], dayfirst=True, errors='coerce')
            df_h = df_h.dropna(subset=['Date']).sort_values('Date')
        except: df_h = pd.DataFrame(columns=hist_cols)
    else:
        df_h = pd.DataFrame(columns=hist_cols)

    return df, df_h

@st.cache_data(ttl=300) # Cache 5 min
def get_live_prices(tickers):
    """Récupération Yahoo Finance"""
    prices = {"CASH": {"cur": 1.0, "prev": 1.0}}
    real_ticks = [t for t in tickers if t != "CASH" and isinstance(t, str)]
    
    if not real_ticks: return prices
    
    try:
        # Optimisation : Téléchargement groupé
        data = yf.download(real_ticks, period="5d", progress=False)
        if 'Close' in data:
            closes = data['Close']
            # Gestion cas unique vs multiple
            if len(real_ticks) == 1:
                 # Séries
                 if len(closes) >= 1:
                     prices[real_ticks[0]] = {"cur": float(closes.iloc[-1]), "prev": float(closes.iloc[-2]) if len(closes)>1 else float(closes.iloc[-1])}
            else:
                # DataFrame
                last_row = closes.iloc[-1]
                prev_row = closes.iloc[-2] if len(closes) > 1 else last_row
                for t in real_ticks:
                    if t in last_row.index:
                        prices[t] = {"cur": float(last_row[t]), "prev": float(prev_row[t])}
    except: pass
    return prices

@st.cache_data(ttl=3600)
def get_market_indices():
    """Surveillance Indices"""
    targets = {"S&P 500": "^GSPC", "CAC 40": "^FCHI", "Bitcoin": "BTC-EUR", "VIX": "^VIX"}
    res = []
    try:
        data = yf.download(list(targets.values()), period="5d", progress=False)['Close']
        for name, tick in targets.items():
            if tick in data.columns:
                cur = data[tick].iloc[-1]
                prev = data[tick].iloc[-2]
                perf = ((cur - prev)/prev)*100
                res.append({"Indice": name, "Prix": cur, "24h %": perf})
    except: pass
    return pd.DataFrame(res)

# --- 4. EXÉCUTION & CALCULS ---

df_pf, df_hist = load_data()
prices = get_live_prices(df_pf['Ticker'].unique())

# Application des prix
df_pf['Prix_Actuel'] = df_pf['Ticker'].apply(lambda t: prices.get(t, {}).get('cur', 0.0) if t != "CASH" else 1.0)
df_pf.loc[df_pf['Prix_Actuel'] == 0, 'Prix_Actuel'] = df_pf['PRU'] # Fallback
df_pf['Prev_Price'] = df_pf['Ticker'].apply(lambda t: prices.get(t, {}).get('prev', 0.0) if t != "CASH" else 1.0)

# KPIs
df_pf['Valo'] = df_pf['Quantité'] * df_pf['Prix_Actuel']
df_pf['Investi'] = df_pf['Quantité'] * df_pf['PRU']
df_pf['PV_Latente'] = df_pf['Valo'] - df_pf['Investi']
df_pf['Perf_%'] = (df_pf['PV_Latente'] / df_pf['Investi'] * 100).fillna(0)
df_pf['Var_24h_€'] = df_pf['Valo'] - (df_pf['Quantité'] * df_pf['Prev_Price'])

TOTAL_ACTUEL = df_pf['Valo'].sum()
CASH_DISPO = df_pf[df_pf['Ticker']=='CASH']['Valo'].sum()
PV_TOTALE = df_pf['PV_Latente'].sum()

# Volatilité vs Historique (Le vrai juge)
if not df_hist.empty:
    last_hist_total = df_hist.iloc[-1]['Total']
    delta_day = TOTAL_ACTUEL - last_hist_total
    delta_pct = (delta_day / last_hist_total) * 100
else:
    delta_day = df_pf['Var_24h_€'].sum()
    delta_pct = 0.0

# --- CALCUL CAGR (Fixe) ---
CAPITAL_INITIAL = 15450.00  
DATE_DEBUT = datetime(2022, 1, 1) 
annees_detention = (datetime.now() - DATE_DEBUT).days / 365.25
if annees_detention > 0 and CAPITAL_INITIAL > 0:
    cagr_val = ((TOTAL_ACTUEL / CAPITAL_INITIAL) ** (1 / annees_detention) - 1) * 100
else:
    cagr_val = 0.0

# --- 5. INTERFACE UTILISATEUR ---

# Header
st.markdown("## 🏛️ Ultimate Liquid Estate")
st.caption(f"Valorisation en temps réel • {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Hero Section
st.markdown(f"""
<div style="background: linear-gradient(135deg, {bg_color} 0%, {card_bg} 100%); 
            padding: 30px; border-radius: 24px; border: 1px solid {border_color}; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center; margin-bottom: 25px;">
    <p style="color: {text_color}; font-size: 14px; text-transform: uppercase; letter-spacing: 2px; margin: 0; opacity: 0.7;">Fortune Nette</p>
    <h1 style="font-size: 64px; margin: 5px 0; background: {metric_gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        {TOTAL_ACTUEL:,.2f} €
    </h1>
    <p style="color: {'#10b981' if delta_day >= 0 else '#ef4444'}; font-weight: 600; font-size: 18px;">
        {delta_day:+.2f} € ({delta_pct:+.2f}%) <span style="color: {text_color}; opacity: 0.5; font-size: 14px;">• Depuis minuit</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Bento Grid (KPIs)
col1, col2, col3 = st.columns(3)
col1.metric("Cash Disponible", f"{CASH_DISPO:,.2f} €", f"{(CASH_DISPO/TOTAL_ACTUEL)*100:.1f}% Alloc.")
col2.metric("Plus-Value Latente", f"{PV_TOTALE:+,.2f} €", f"{(PV_TOTALE/(TOTAL_ACTUEL-PV_TOTALE))*100:.2f}%")
col3.metric("CAGR (Annuel)", f"{cagr_val:.2f} %", f"Depuis {DATE_DEBUT.year}")

# --- SECTION 1 : PORTEFEUILLE ---
st.markdown("<div class='section-header'>📋 Détail du Portefeuille</div>", unsafe_allow_html=True)

st.dataframe(
    df_pf[['Nom', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo', 'Perf_%', 'Var_24h_€']],
    hide_index=True,
    use_container_width=True,
    column_config={
        "Nom": st.column_config.TextColumn("Actif", width="medium"),
        "Valo": st.column_config.NumberColumn("Valorisation", format="%.2f €"),
        "PRU": st.column_config.NumberColumn("Prix Revient", format="%.2f €"),
        "Prix_Actuel": st.column_config.NumberColumn("Cours", format="%.2f €"),
        "Perf_%": st.column_config.ProgressColumn("Perf %", format="%.2f %%", min_value=-20, max_value=20),
        "Var_24h_€": st.column_config.NumberColumn("24h", format="%+.2f €")
    }
)

st.markdown("---")

# --- SECTION 2 : ANALYSE GRAPHIQUE ---
st.markdown("<div class='section-header'>📊 Analyse & Marché</div>", unsafe_allow_html=True)

if not df_hist.empty:
    c1, c2 = st.columns(2)
    
    with c1:
        st.caption("Trajectoire Patrimoniale")
        fig = px.area(df_hist, x='Date', y='Total', line_shape='spline')
        fig.update_layout(
            template="plotly_dark" if dark_mode else "simple_white", # Thème natif Plotly
            margin=dict(l=0,r=0,t=10,b=0), height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_traces(line_color=chart_line_color, fillcolor=chart_fill_color)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.caption("Benchmark (Base 100)")
        if 'PF_Index100' in df_hist.columns and 'ESE_Index100' in df_hist.columns:
            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['PF_Index100'], name="Moi", line=dict(color=text_color, width=2)))
            fig_b.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['ESE_Index100'], name="S&P500", line=dict(color='#94a3b8', dash='dot')))
            fig_b.update_layout(
                template="plotly_dark" if dark_mode else "simple_white",
                margin=dict(l=0,r=0,t=10,b=0), height=350, legend=dict(orientation="h", y=1.1, x=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_b, use_container_width=True)
else:
    st.info("Attente de la première exécution du Robot de nuit...")

# Indices Marché
st.caption("Pulsation Mondiale")
df_m = get_market_indices()
if not df_m.empty:
    cols = st.columns(len(df_m))
    for i, row in df_m.iterrows():
        with cols[i]:
            st.metric(row['Indice'], f"{row['Prix']:.2f}", f"{row['24h %']:+.2f}%")

st.markdown("---")

# --- SECTION 3 : PROJECTION ---
st.markdown("<div class='section-header'>🔮 Projection & Rente</div>", unsafe_allow_html=True)

col_sim_input, col_sim_graph = st.columns([1, 3])

with col_sim_input:
    with st.expander("Paramètres de Simulation", expanded=True):
        apport_mensuel = st.number_input("Apport Mensuel (€)", value=1000, step=100, key="sim_add")
        taux_annuel = st.slider("Rendement Annuel (%)", 2.0, 15.0, 8.0, 0.5, key="sim_rate")
        duree_ans = st.slider("Horizon (Années)", 5, 30, 15, key="sim_years")
        
        st.markdown(f"""
        <div style="margin-top: 20px; padding: 15px; background: {card_bg}; border-radius: 12px; border: 1px solid {border_color};">
            <small style="color: {text_color}; opacity: 0.7;">Capital Départ</small><br>
            <strong style="color: {text_color};">{TOTAL_ACTUEL:,.0f} €</strong>
        </div>
        """, unsafe_allow_html=True)

with col_sim_graph:
    # Calcul Simulation
    annees = range(datetime.now().year, datetime.now().year + duree_ans + 1)
    capital = [TOTAL_ACTUEL]
    for i in range(duree_ans):
        nouveau_montant = (capital[-1] + (apport_mensuel * 12)) * (1 + taux_annuel/100)
        capital.append(nouveau_montant)
    
    df_sim = pd.DataFrame({"Année": annees, "Capital": capital})
    
    # Graphique Projection
    fig_sim = px.area(df_sim, x="Année", y="Capital")
    fig_sim.update_layout(
        template="plotly_dark" if dark_mode else "simple_white",
        height=400,
        margin=dict(l=0,r=0,t=20,b=0),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    fig_sim.update_traces(line_color='#10b981', fillcolor='rgba(16, 185, 129, 0.15)')
    fig_sim.add_hline(y=1000000, line_dash="dot", line_color="#cbd5e1", annotation_text="1M€ (Liberté)", annotation_position="top left")
    
    st.plotly_chart(fig_sim, use_container_width=True)
    
    final_cap = capital[-1]
    k1, k2, k3 = st.columns(3)
    k1.metric("Capital Final", f"{final_cap:,.0f} €")
    k2.metric("Total Intérêts", f"{(final_cap - TOTAL_ACTUEL - (apport_mensuel * 12 * duree_ans)):,.0f} €")
    k3.metric("Rente Mensuelle (4%)", f"{(final_cap * 0.04 / 12):,.0f} €", "Revenu Passif")