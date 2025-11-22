import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- IMPORT DES FONCTIONS UTILITAIRES ---
from utils import load_data, get_live_prices, get_market_indices, create_bento_card

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tableau de Bord", layout="wide", page_icon="💎")

# --- 2. GESTION DU THÈME (DARK / LIGHT) ---
with st.sidebar:
    st.header("⚙️ Préférences")
    dark_mode = st.toggle("🌙 Mode Sombre", value=True)
    st.caption("Tableau de Bord V.1.7 (Refactored)")

if dark_mode:
    bg_color = "#0f172a"
    text_color = "#f8fafc"
    card_bg = "rgba(30, 41, 59, 0.3)" 
    border_color = "rgba(255, 255, 255, 0.05)"
    chart_line_color = "#38bdf8"
    chart_fill_color = "rgba(56, 189, 248, 0.15)"
    metric_gradient = "linear-gradient(135deg, #38bdf8 0%, #818cf8 100%)"
    css_theme = """
    .stApp { background-color: #020617; color: #f8fafc; }
    h1, h2, h3, p, span, div { color: #f8fafc; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    """
else:
    bg_color = "#f0f4f8"
    text_color = "#1e293b"
    card_bg = "rgba(255, 255, 255, 0.3)"
    border_color = "rgba(255, 255, 255, 0.4)"
    chart_line_color = "#2563eb"
    chart_fill_color = "rgba(37, 99, 235, 0.1)"
    metric_gradient = "linear-gradient(135deg, #0f172a 0%, #334155 100%)"
    css_theme = """
    .stApp { background-color: #f0f4f8; color: #1e293b; }
    h1, h2, h3 { color: #0f172a; }
    div[data-testid="stMetricLabel"] { color: #64748b !important; }
    """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');
    .stApp {{ font-family: 'Outfit', sans-serif; }}
    {css_theme}
    div[data-testid="stMetric"], div.stPlotlyChart, div.stExpander {{
        background: {card_bg} !important;
        backdrop-filter: blur(15px);
        border-radius: 24px;
        border: 1px solid {border_color};
        padding: 24px !important;
    }}
    div[data-testid="stDataFrame"] {{ background: transparent !important; border: none !important; }}
    div[data-testid="stMetricValue"] {{
        font-size: 32px; font-weight: 800;
        background: {metric_gradient};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .section-header {{
        margin-top: 40px; margin-bottom: 20px; font-size: 24px; font-weight: 700;
        border-bottom: 2px solid {border_color}; padding-bottom: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# --- 3. EXÉCUTION (VIA UTILS) ---

df_pf, df_hist = load_data()

if df_pf.empty:
    st.warning("⚠️ Portefeuille vide ou non chargé.")
    TOTAL_ACTUEL = 0.0
    CASH_DISPO = 0.0
    PV_TOTALE = 0.0
    delta_day = 0.0
    delta_pct = 0.0
else:
    prices = get_live_prices(df_pf['Ticker'].unique())
    
    df_pf['Prix_Actuel'] = df_pf['Ticker'].apply(lambda t: prices.get(t, {}).get('cur', 0.0) if t != "CASH" else 1.0)
    df_pf.loc[(df_pf['Prix_Actuel'] == 0) & (df_pf['Ticker'] != "CASH"), 'Prix_Actuel'] = df_pf['PRU']
    df_pf['Prev_Price'] = df_pf['Ticker'].apply(lambda t: prices.get(t, {}).get('prev', 0.0) if t != "CASH" else 1.0)
    
    df_pf['Valo'] = df_pf['Quantité'] * df_pf['Prix_Actuel']
    df_pf['Investi'] = df_pf['Quantité'] * df_pf['PRU']
    df_pf['PV_Latente'] = df_pf['Valo'] - df_pf['Investi']
    df_pf['Perf_%'] = (df_pf['PV_Latente'] / df_pf['Investi'] * 100).fillna(0)
    df_pf['Var_24h_€'] = df_pf['Valo'] - (df_pf['Quantité'] * df_pf['Prev_Price'])

    TOTAL_ACTUEL = df_pf['Valo'].sum()
    CASH_DISPO = df_pf[df_pf['Ticker']=='CASH']['Valo'].sum()
    PV_TOTALE = df_pf['PV_Latente'].sum()

    # Calcul Delta
    delta_day = df_pf['Var_24h_€'].sum()
    delta_pct = 0.0
    if not df_hist.empty:
        try:
            last_val = df_hist.iloc[-1]['Total']
            if last_val > 0:
                delta_day = TOTAL_ACTUEL - last_val
                delta_pct = (delta_day / last_val) * 100
        except: pass

CAPITAL_INITIAL = 15450.00  
DATE_DEBUT = datetime(2022, 1, 1) 
annees = (datetime.now() - DATE_DEBUT).days / 365.25
cagr_val = ((TOTAL_ACTUEL / CAPITAL_INITIAL) ** (1 / annees) - 1) * 100 if annees > 0 and TOTAL_ACTUEL > 0 else 0.0

# --- 4. AFFICHAGE ---

st.markdown("## 🏛️ Tableau de Bord")
st.caption(f"Dernière synchro • {datetime.now().strftime('%d/%m/%Y %H:%M')}")

st.markdown(f"""
<div style="background: linear-gradient(135deg, {bg_color} 0%, {card_bg} 100%); 
            padding: 30px; border-radius: 24px; border: 1px solid {border_color}; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center; margin-bottom: 25px;">
    <p style="color: {text_color}; font-size: 14px; text-transform: uppercase; letter-spacing: 2px; margin: 0; opacity: 0.7;">Portefeuille</p>
    <h1 style="font-size: 64px; margin: 5px 0; background: {metric_gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        {TOTAL_ACTUEL:,.2f} €
    </h1>
    <p style="color: {'#10b981' if delta_day >= -0.01 else '#ef4444'}; font-weight: 600; font-size: 18px;">
        {delta_day:+.2f} € ({delta_pct:+.2f}%) <span style="color: {text_color}; opacity: 0.5; font-size: 14px;">• sur 24h</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Calcul Volatilité
if not df_hist.empty:
    def clean_pct_metric(x):
        try: return float(str(x).replace('%', '').replace(',', '.')) / 100
        except: return 0.0
    daily_rets = df_hist['PF_Return_TWR'].apply(clean_pct_metric)
    volatility = daily_rets.std() * (252 ** 0.5) * 100 
else:
    volatility = 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Liquidité", f"{CASH_DISPO:,.2f} €", f"{(CASH_DISPO/TOTAL_ACTUEL)*100:.1f}% Alloc." if TOTAL_ACTUEL > 0 else "0%")
c2.metric("Plus-Value Latente", f"{PV_TOTALE:+,.2f} €", f"{(PV_TOTALE/(TOTAL_ACTUEL-PV_TOTALE))*100:.2f}%" if (TOTAL_ACTUEL-PV_TOTALE)!=0 else "0%")
c3.metric("CAGR (Annuel)", f"{cagr_val:.2f} %", f"Depuis {DATE_DEBUT.year}")
c4.metric("Volatilité", f"{volatility:.2f} %", "Risque Annualisé")

# --- GRAPHIQUES ---
st.markdown("---")
col_titre, col_filtre = st.columns([3, 1])
with col_titre:
    st.markdown("<div class='section-header'>📊 Analyse & Marché</div>", unsafe_allow_html=True)
with col_filtre:
    periode = st.selectbox("Période", ["Tout", "YTD (Année)", "1 An", "6 Mois", "3 Mois"], index=1)

# 1. Application du Filtre Temporel
if not df_hist.empty:
    df_filtered = df_hist.copy()
    today = datetime.now()
    
    if periode == "YTD (Année)":
        start_date = datetime(today.year, 1, 1)
        df_filtered = df_filtered[df_filtered['Date'] >= start_date]
    elif periode == "1 An":
        start_date = today - pd.DateOffset(years=1)
        df_filtered = df_filtered[df_filtered['Date'] >= start_date]
    elif periode == "6 Mois":
        start_date = today - pd.DateOffset(months=6)
        df_filtered = df_filtered[df_filtered['Date'] >= start_date]
    elif periode == "3 Mois":
        start_date = today - pd.DateOffset(months=3)
        df_filtered = df_filtered[df_filtered['Date'] >= start_date]

    # 2. Calcul du Drawdown sur la période filtrée
    if not df_filtered.empty:
        max_histo = df_filtered['Total'].max()
        # Attention : on prend la dernière valeur de la période filtrée
        val_actuelle = df_filtered.iloc[-1]['Total']
        drawdown = ((val_actuelle - max_histo) / max_histo) * 100
        
        col_dd1, col_dd2 = st.columns([1, 3])
        with col_dd1:
            st.metric("Drawdown (Période)", f"{drawdown:.2f} %", delta_color="off")
            st.caption(f"Plus Haut Période : {max_histo:,.0f} €")
        
        with col_dd2:
            if drawdown > -5: st.info("💎 **Solidité :** Proche du sommet de la période.")
            elif drawdown > -15: st.warning("⚠️ **Correction :** Le marché respire.")
            else: st.error("🚨 **Zone de baisse :** Opportunité potentielle.")

    # 3. Affichage des Graphiques (Connectés à df_filtered !)
    if not df_filtered.empty and len(df_filtered) > 1:
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Trajectoire ({periode})")
            # CORRECTION ICI : On utilise df_filtered
            fig = px.area(df_filtered, x='Date', y='Total', line_shape='spline')
            fig.update_layout(
                template="plotly_dark" if dark_mode else "simple_white", 
                margin=dict(l=0,r=0,t=10,b=0), height=350, 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_traces(line_color=chart_line_color, fillcolor=chart_fill_color)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.caption(f"Benchmark Base 100 ({periode})")
            # CORRECTION ICI : On utilise df_filtered
            if 'PF_Index100' in df_filtered.columns and 'ESE_Index100' in df_filtered.columns:
                # Recalcul de la base 100 pour que les courbes partent du même point sur le graph zoomé
                first_pf = df_filtered.iloc[0]['PF_Index100']
                first_ese = df_filtered.iloc[0]['ESE_Index100']
                
                # Normalisation dynamique pour le graphique
                y_moi = (df_filtered['PF_Index100'] / first_pf) * 100
                y_ese = (df_filtered['ESE_Index100'] / first_ese) * 100

                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=df_filtered['Date'], y=y_moi, name="Moi", line=dict(color=text_color, width=2)))
                fig_b.add_trace(go.Scatter(x=df_filtered['Date'], y=y_ese, name="S&P500", line=dict(color='#94a3b8', dash='dot')))
                fig_b.update_layout(
                    template="plotly_dark" if dark_mode else "simple_white", 
                    margin=dict(l=0,r=0,t=10,b=0), height=350, 
                    legend=dict(orientation="h", y=1.1, x=0), 
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.warning("Pas assez de données pour cette période.")

# --- MARCHÉS & ALLOCATION ---
st.caption("Pulsation Mondiale")
df_m = get_market_indices()
if not df_m.empty:
    cols = st.columns(len(df_m))
    for i, row in df_m.iterrows():
        with cols[i]:
            st.metric(row['Indice'], f"{row['Prix']:.2f}", f"{row['24h %']:+.2f}%")

fig_alloc = px.pie(df_pf, values='Valo', names='Nom', title='Répartition', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel if not dark_mode else px.colors.qualitative.Bold)
fig_alloc.update_layout(
    template="plotly_dark" if dark_mode else "plotly_white", 
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
    showlegend=True, legend=dict(orientation="v", y=0.5, x=1.02, xanchor="left", yanchor="middle"), 
    margin=dict(t=30, b=30, l=20, r=200)
)
st.plotly_chart(fig_alloc, use_container_width=True)

# --- DETAIL (BENTO) ---
st.markdown("---")
st.markdown("<div class='section-header'>📋 Détail des Actifs</div>", unsafe_allow_html=True)

if not df_pf.empty:
    COLS = 3
    rows = [df_pf.iloc[i:i + COLS] for i in range(0, len(df_pf), COLS)]
    for row_data in rows:
        cols = st.columns(COLS)
        for i, (index, asset) in enumerate(row_data.iterrows()):
            with cols[i]:
                html_card = create_bento_card(asset, card_bg, border_color, text_color, metric_gradient)
                st.markdown(html_card.strip(), unsafe_allow_html=True)
else:
    st.info("Aucun actif.")

# --- PERFORMANCE MENSUELLE ---
if not df_hist.empty:
    st.markdown("<div class='section-header'>📅 Performance Mensuelle</div>", unsafe_allow_html=True)
    
    df_matrix = df_hist.copy()
    df_matrix['Year'] = df_matrix['Date'].dt.year
    df_matrix['Month'] = df_matrix['Date'].dt.month
    
    def clean_pct(x):
        if isinstance(x, str): return float(x.replace('%', '').replace(',', '.')) / 100
        return 0.0
    df_matrix['Daily_Return'] = df_matrix['PF_Return_TWR'].apply(clean_pct)
    
    monthly_returns = df_matrix.groupby(['Year', 'Month'])['Daily_Return'].apply(lambda x: (1 + x).prod() - 1) * 100
    matrix = monthly_returns.unstack(level=1).fillna(0)
    
    months_map = {1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Juin', 7: 'Juil', 8: 'Août', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'}
    matrix.columns = [months_map.get(c, c) for c in matrix.columns]
    
    fig_heat = px.imshow(matrix, text_auto='.2f', aspect="auto", color_continuous_scale="RdBu", color_continuous_midpoint=0)
    fig_heat.update_layout(template="plotly_dark" if dark_mode else "plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title=None, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)