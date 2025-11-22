import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- IMPORT DES FONCTIONS UTILITAIRES ---
from utils import load_data, get_live_prices, get_market_indices, create_bento_card

# --- 1. CONFIGURATION & THÈME (100% DARK MODE) ---
st.set_page_config(page_title="Tableau de Bord", layout="wide", page_icon="💎")

# Définition des couleurs fixes (Dark Mode)
BG_COLOR = "#0f172a"
TEXT_COLOR = "#f8fafc"
CARD_BG = "rgba(30, 41, 59, 0.3)" 
BORDER_COLOR = "rgba(255, 255, 255, 0.05)"
CHART_LINE_COLOR = "#38bdf8"
CHART_FILL_COLOR = "rgba(56, 189, 248, 0.15)"
METRIC_GRADIENT = "linear-gradient(135deg, #38bdf8 0%, #818cf8 100%)"

# Injection du CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');
    .stApp {{ 
        background-color: {BG_COLOR}; 
        color: {TEXT_COLOR}; 
        font-family: 'Outfit', sans-serif; 
    }}
    h1, h2, h3, p, span, div {{ color: {TEXT_COLOR}; }}
    div[data-testid="stMetricLabel"] {{ color: #94a3b8 !important; }}
    
    /* Styles des cartes et graphiques */
    div[data-testid="stMetric"], div.stPlotlyChart, div.stExpander {{
        background: {CARD_BG} !important;
        backdrop-filter: blur(15px);
        border-radius: 24px;
        border: 1px solid {BORDER_COLOR};
        padding: 24px !important;
    }}
    div[data-testid="stDataFrame"] {{ background: transparent !important; border: none !important; }}
    div[data-testid="stMetricValue"] {{
        font-size: 32px; font-weight: 800;
        background: {METRIC_GRADIENT};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .section-header {{
        margin-top: 40px; margin-bottom: 20px; font-size: 24px; font-weight: 700;
        border-bottom: 2px solid {BORDER_COLOR}; padding-bottom: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# --- 2. EXÉCUTION (VIA UTILS) ---

df_pf, df_hist = load_data()

if df_pf.empty:
    st.warning("⚠️ Portefeuille vide ou non chargé.")
    TOTAL_ACTUEL, CASH_DISPO, PV_TOTALE = 0.0, 0.0, 0.0
    delta_day, delta_pct = 0.0, 0.0
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

# --- 3. AFFICHAGE ---

with st.sidebar:
    st.header("💎 Ultimate Estate")
    st.caption("Version 2.0 (Dark Only)")
    st.info(f"📅 **{datetime.now().strftime('%d/%m/%Y')}**")

st.markdown(f"""
<div style="background: linear-gradient(135deg, {BG_COLOR} 0%, {CARD_BG} 100%); 
            padding: 30px; border-radius: 24px; border: 1px solid {BORDER_COLOR}; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center; margin-bottom: 25px;">
    <p style="color: {TEXT_COLOR}; font-size: 14px; text-transform: uppercase; letter-spacing: 2px; margin: 0; opacity: 0.7;">Portefeuille</p>
    <h1 style="font-size: 64px; margin: 5px 0; background: {METRIC_GRADIENT}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        {TOTAL_ACTUEL:,.2f} €
    </h1>
    <p style="color: {'#10b981' if delta_day >= -0.01 else '#ef4444'}; font-weight: 600; font-size: 18px;">
        {delta_day:+.2f} € ({delta_pct:+.2f}%) <span style="color: {TEXT_COLOR}; opacity: 0.5; font-size: 14px;">• sur 24h</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Calcul Indicateurs de Risque (Volatilité & Sharpe)
if not df_hist.empty:
    def clean_pct_metric(x):
        try: return float(str(x).replace('%', '').replace(',', '.')) / 100
        except: return 0.0
    
    daily_rets = df_hist['PF_Return_TWR'].apply(clean_pct_metric)
    
    # Volatilité Annualisée
    volatility = daily_rets.std() * (252 ** 0.5) * 100 
    
    # Ratio de Sharpe (Taux sans risque supposé 3% = 0.03)
    mean_ret = daily_rets.mean() * 252
    risk_free_rate = 0.03 
    sharpe_ratio = (mean_ret - risk_free_rate) / (volatility/100) if volatility > 0 else 0.0
else:
    volatility = 0.0
    sharpe_ratio = 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Liquidité", f"{CASH_DISPO:,.2f} €", f"{(CASH_DISPO/TOTAL_ACTUEL)*100:.1f}% Alloc." if TOTAL_ACTUEL > 0 else "0%")
c2.metric("CAGR (Annuel)", f"{cagr_val:.2f} %", f"Sharpe: {sharpe_ratio:.2f}") # J'ai mis le Sharpe ici en petit
c3.metric("Plus-Value", f"{PV_TOTALE:+,.2f} €", f"{(PV_TOTALE/(TOTAL_ACTUEL-PV_TOTALE))*100:.2f}%")
c4.metric("Volatilité", f"{volatility:.2f} %", "Risque Annualisé")

# --- GRAPHIQUES (MODULE ANALYSTE) ---
st.markdown("---")
col_titre, col_filtre = st.columns([3, 1])
with col_titre:
    st.markdown("<div class='section-header'>📊 Analyse & Marché</div>", unsafe_allow_html=True)
with col_filtre:
    periode = st.selectbox("Période", ["Tout", "YTD (Année)", "1 An", "6 Mois", "3 Mois"], index=1)

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

    if not df_filtered.empty:
        max_histo = df_filtered['Total'].max()
        val_actuelle = df_filtered.iloc[-1]['Total']
        drawdown = ((val_actuelle - max_histo) / max_histo) * 100
        
        col_dd1, col_dd2 = st.columns([1, 3])
        with col_dd1:
            st.metric("Drawdown (Période)", f"{drawdown:.2f} %", delta_color="off")
            st.caption(f"Plus Haut Période : {max_histo:,.0f} €")
        
        with col_dd2:
            if drawdown > -5: st.info("💎 **Solidité :** Proche du sommet.")
            elif drawdown > -15: st.warning("⚠️ **Correction :** Le marché respire.")
            else: st.error("🚨 **Zone de baisse :** Opportunité potentielle.")

    if not df_filtered.empty and len(df_filtered) > 1:
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Trajectoire ({periode})")
            fig = px.area(df_filtered, x='Date', y='Total', line_shape='spline')
            fig.update_layout(
                template="plotly_dark", # DARK FORCÉ
                margin=dict(l=0,r=0,t=10,b=0), height=350, 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_traces(line_color=CHART_LINE_COLOR, fillcolor=CHART_FILL_COLOR)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.caption(f"Benchmark Base 100 ({periode})")
            if 'PF_Index100' in df_filtered.columns and 'ESE_Index100' in df_filtered.columns:
                first_pf = df_filtered.iloc[0]['PF_Index100']
                first_ese = df_filtered.iloc[0]['ESE_Index100']
                
                y_moi = (df_filtered['PF_Index100'] / first_pf) * 100
                y_ese = (df_filtered['ESE_Index100'] / first_ese) * 100

                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=df_filtered['Date'], y=y_moi, name="Moi", line=dict(color=TEXT_COLOR, width=2)))
                fig_b.add_trace(go.Scatter(x=df_filtered['Date'], y=y_ese, name="S&P500", line=dict(color='#94a3b8', dash='dot')))
                fig_b.update_layout(
                    template="plotly_dark", # DARK FORCÉ
                    margin=dict(l=0,r=0,t=10,b=0), height=350, 
                    legend=dict(orientation="h", y=1.1, x=0), 
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.warning("Pas assez de données pour cette période.")

# --- MARCHÉS & ALLOCATION (MODULE STRATÈGE) ---
st.caption("Pulsation Mondiale")
df_m = get_market_indices()
if not df_m.empty:
    cols = st.columns(len(df_m))
    for i, row in df_m.iterrows():
        with cols[i]:
            st.metric(row['Indice'], f"{row['Prix']:.2f}", f"{row['24h %']:+.2f}%")

st.markdown("---")
col_alloc_titre, col_alloc_option = st.columns([3, 1])
with col_alloc_titre:
    st.markdown("<div class='section-header'>🍰 Allocation d'Actifs</div>", unsafe_allow_html=True)
with col_alloc_option:
    # AMÉLIORATION : Toggle pour changer de vue
    vue_alloc = st.radio("Vue", ["Actifs", "Types"], horizontal=True, label_visibility="collapsed")

col_to_plot = 'Nom' if vue_alloc == "Actifs" else 'Type'

fig_alloc = px.pie(
    df_pf, 
    values='Valo', 
    names=col_to_plot, 
    title=f'Répartition par {col_to_plot}', 
    hole=0.6, 
    color_discrete_sequence=px.colors.qualitative.Bold # Couleurs vives pour Dark Mode
)
fig_alloc.update_layout(
    template="plotly_dark", # DARK FORCÉ
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
                html_card = create_bento_card(asset, CARD_BG, BORDER_COLOR, TEXT_COLOR, METRIC_GRADIENT)
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
    fig_heat.update_layout(
        template="plotly_dark", # DARK FORCÉ
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        xaxis_title=None, yaxis_title=None, margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- ANALYSE DE CORRÉLATION ---
st.markdown("---")
st.markdown("<div class='section-header'>🔗 Corrélation (Moi vs Marché)</div>", unsafe_allow_html=True)

if not df_hist.empty:
    # On prépare deux séries temporelles propres
    s_moi = df_hist['PF_Return_TWR'].apply(clean_pct)
    s_ese = df_hist['ESE_Return'].apply(clean_pct)
    
    # Création d'un DataFrame conjoint
    df_corr = pd.DataFrame({"Mon Portefeuille": s_moi, "S&P 500 (ESE)": s_ese})
    
    # Calcul de la corrélation (Rolling Window 30 jours pour voir l'évolution)
    rolling_corr = df_corr['Mon Portefeuille'].rolling(window=30).corr(df_corr['S&P 500 (ESE)'])
    
    # Graphique
    fig_corr = px.line(x=df_hist['Date'], y=rolling_corr)
    fig_corr.update_traces(line_color="#f59e0b", name="Corrélation (30j)") # Couleur Ambre
    fig_corr.add_hline(y=1, line_dash="dot", annotation_text="Corrélation Parfaite", annotation_position="bottom right")
    fig_corr.add_hline(y=0, line_dash="dot", annotation_text="Décorrelé", annotation_position="bottom right")
    
    fig_corr.update_layout(
        title="Évolution de la corrélation avec le S&P 500 (Glissant 30j)",
        yaxis_title="Coefficient (-1 à 1)",
        xaxis_title=None,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    curr_corr = df_corr.corr().iloc[0, 1]
    st.info(f"🔗 **Corrélation Globale : {curr_corr:.2f}**. " + 
            ("Vous suivez le marché." if curr_corr > 0.7 else "Votre portefeuille est bien diversifié/décorrelé du S&P500."))