import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# --- IMPORT DES FONCTIONS UTILITAIRES ---
# On conserve votre structure robuste existante
from utils import load_data, get_live_prices, get_market_indices

# --- 1. CONFIGURATION & THÈME "ULTIMATE ESTATE" ---
st.set_page_config(page_title="Ultimate Estate", layout="wide", page_icon="🏛️")

# Palette de couleurs "Private Banking"
COLOR_BG = "#0f172a"        # Slate 900
COLOR_CARD = "#1e293b"      # Slate 800
COLOR_TEXT = "#f1f5f9"      # Slate 100
COLOR_ACCENT = "#fbbf24"    # Amber 400 (Or)
COLOR_POS = "#34d399"       # Emerald 400
COLOR_NEG = "#f87171"       # Red 400
COLOR_CHART = "#38bdf8"     # Sky 400

# CSS Avancé
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600;800&display=swap');
    
    .stApp {{ 
        background-color: {COLOR_BG}; 
        color: {COLOR_TEXT}; 
        font-family: 'Manrope', sans-serif; 
    }}
    
    /* Suppression des marges par défaut pour un look dense */
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    
    /* Cards Modernes */
    div.stMetric, div.stPlotlyChart, div[data-testid="stDataFrame"] {{
        background-color: {COLOR_CARD};
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    
    /* Titres */
    h1, h2, h3 {{ font-family: 'Manrope', sans-serif; font-weight: 800; letter-spacing: -0.5px; }}
    h1 {{ color: {COLOR_ACCENT}; }}
    
    /* Metrics */
    div[data-testid="stMetricLabel"] {{ font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.8rem; font-weight: 700; color: {COLOR_TEXT}; }}
    div[data-testid="stMetricDelta"] {{ font-size: 0.9rem; }}
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; background-color: transparent; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #94a3b8;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {COLOR_ACCENT};
        border-bottom: 2px solid {COLOR_ACCENT};
    }}
</style>
""", unsafe_allow_html=True)

# --- 2. CHARGEMENT DES DONNÉES ---
df_pf, df_hist = load_data()

# Initialisation des variables globales
TOTAL_ACTUEL = 0.0
CASH_DISPO = 0.0
PV_TOTALE = 0.0
delta_day_abs = 0.0
delta_day_pct = 0.0
sharpe_ratio = 0.0
volatility = 0.0
cagr_val = 0.0

if not df_pf.empty:
    # Récupération Prix
    prices = get_live_prices(df_pf['Ticker'].unique())
    
    # Calculs Portefeuille
    df_pf['Prix_Actuel'] = df_pf['Ticker'].apply(lambda t: prices.get(t, {}).get('cur', 0.0) if t != "CASH" else 1.0)
    # Fallback PRU si prix nul
    df_pf.loc[(df_pf['Prix_Actuel'] == 0) & (df_pf['Ticker'] != "CASH"), 'Prix_Actuel'] = df_pf['PRU']
    df_pf['Prev_Price'] = df_pf['Ticker'].apply(lambda t: prices.get(t, {}).get('prev', 0.0) if t != "CASH" else 1.0)
    
    df_pf['Valo'] = df_pf['Quantité'] * df_pf['Prix_Actuel']
    df_pf['Investi'] = df_pf['Quantité'] * df_pf['PRU']
    df_pf['PV_Latente'] = df_pf['Valo'] - df_pf['Investi']
    df_pf['Perf_%'] = (df_pf['PV_Latente'] / df_pf['Investi'] * 100).fillna(0)
    df_pf['Poids_%'] = (df_pf['Valo'] / df_pf['Valo'].sum() * 100).fillna(0)
    
    # KPI Globaux
    TOTAL_ACTUEL = df_pf['Valo'].sum()
    CASH_DISPO = df_pf[df_pf['Ticker']=='CASH']['Valo'].sum()
    PV_TOTALE = df_pf['PV_Latente'].sum()
    
    # Variation Jour (Calcul précis basé sur historique si dispo, sinon estimation live)
    if not df_hist.empty:
        try:
            last_close = df_hist.iloc[-1]['Total']
            delta_day_abs = TOTAL_ACTUEL - last_close
            delta_day_pct = (delta_day_abs / last_close) * 100 if last_close != 0 else 0
        except: pass
    
    # Calculs CAGR & Risque
    CAPITAL_INITIAL = 15450.00  # À ajuster si vos apports changent
    DATE_DEBUT = datetime(2022, 1, 1) 
    annees = (datetime.now() - DATE_DEBUT).days / 365.25
    if annees > 0 and TOTAL_ACTUEL > 0:
        cagr_val = ((TOTAL_ACTUEL / CAPITAL_INITIAL) ** (1 / annees) - 1) * 100
        
    if not df_hist.empty:
        def clean_pct_metric(x):
            try: return float(str(x).replace('%', '').replace(',', '.')) / 100
            except: return 0.0
        daily_rets = df_hist['PF_Return_TWR'].apply(clean_pct_metric)
        volatility = daily_rets.std() * (252 ** 0.5) * 100
        mean_ret = daily_rets.mean() * 252
        sharpe_ratio = (mean_ret - 0.03) / (volatility/100) if volatility > 0 else 0.0

# --- 3. INTERFACE UTILISATEUR ---

# HEADER / TOP BAR
c1, c2 = st.columns([3, 1])
with c1:
    st.title("Patrimoine")
    st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d %B %Y à %H:%M')}")
with c2:
    # Un petit indicateur de marché global rapide
    df_m = get_market_indices()
    if not df_m.empty:
        sp500 = df_m[df_m['Indice'] == 'S&P 500']['24h %'].values
        val_sp = sp500[0] if len(sp500) > 0 else 0
        color_sp = COLOR_POS if val_sp >= 0 else COLOR_NEG
        st.markdown(f"""
        <div style="text-align: right; padding: 10px;">
            <span style="color: #94a3b8; font-size: 0.8rem;">TENDANCE S&P 500</span><br>
            <span style="color: {color_sp}; font-weight: 800; font-size: 1.2rem;">{val_sp:+.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

# METRICS ROW (KPI)
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Valorisation Nette", f"{TOTAL_ACTUEL:,.2f} €", f"{delta_day_abs:+.2f} €")
kpi2.metric("Variation (24h)", f"{delta_day_pct:+.2f} %", delta_color="normal")
kpi3.metric("Plus-Value Latente", f"{PV_TOTALE:+,.2f} €", f"{(PV_TOTALE/(TOTAL_ACTUEL-PV_TOTALE)*100):.2f} %")
kpi4.metric("Cash Disponible", f"{CASH_DISPO:,.2f} €", f"{(CASH_DISPO/TOTAL_ACTUEL)*100:.1f} % Alloc.")
kpi5.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}", "Risque ajusté")

st.markdown("---")

# ONGLETS PRINCIPAUX
tab_synthese, tab_perf, tab_alloc = st.tabs(["🏢 Synthèse & Marché", "📈 Analyse Performance", "💼 Allocation & Actifs"])

# --- ONGLET 1 : SYNTHÈSE ---
with tab_synthese:
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("Trajectoire du Patrimoine")
        
        if not df_hist.empty:
            # Filtres temporels simples
            periode = st.selectbox("Horizon", ["YTD", "1 An", "Tout"], index=0, label_visibility="collapsed")
            
            df_chart = df_hist.copy()
            today = datetime.now()
            if periode == "YTD":
                df_chart = df_chart[df_chart['Date'] >= datetime(today.year, 1, 1)]
            elif periode == "1 An":
                df_chart = df_chart[df_chart['Date'] >= today - pd.DateOffset(years=1)]

            # Graphique Principal (Area Chart)
            fig = px.area(df_chart, x='Date', y='Total', height=400)
            fig.update_traces(line_color=COLOR_CHART, fillcolor="rgba(56, 189, 248, 0.1)")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0,r=0,t=0,b=0),
                xaxis_title=None, yaxis_title=None,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("En attente de données historiques...")

    with col_side:
        st.subheader("Pulsation Marché")
        if not df_m.empty:
            for index, row in df_m.iterrows():
                val = row['24h %']
                col_val = COLOR_POS if val >= 0 else COLOR_NEG
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <span style="font-weight: 600;">{row['Indice']}</span>
                    <span style="color: {col_val}; font-weight: 700;">{val:+.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("CAGR (Taux Actuariel)", f"{cagr_val:.2f} %", "Croissance Moyenne/An")
        st.metric("Volatilité", f"{volatility:.2f} %", "Risque Annualisé")

# --- ONGLET 2 : PERFORMANCE & RISQUE ---
with tab_perf:
    if not df_hist.empty:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Benchmark : Moi vs S&P 500")
            # Normalisation base 100 dynamique selon la période affichée précédemment ou défaut
            df_bench = df_hist.copy()
            
            # Calcul base 100
            first_pf = df_bench['Total'].iloc[0]
            # On essaie de récupérer l'index S&P500 recalculé ou on le recrée
            # Ici on simplifie en prenant les colonnes Index100 si elles existent bien
            if 'PF_Index100' in df_bench.columns and 'ESE_Index100' in df_bench.columns:
                # Rebaser à 100 sur le début de la sélection pour comparaison juste
                base_pf = df_bench.iloc[0]['PF_Index100']
                base_ese = df_bench.iloc[0]['ESE_Index100']
                
                df_bench['Moi_Rebased'] = (df_bench['PF_Index100'] / base_pf) * 100
                df_bench['SP500_Rebased'] = (df_bench['ESE_Index100'] / base_ese) * 100
                
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=df_bench['Date'], y=df_bench['Moi_Rebased'], name="Mon Portefeuille", line=dict(color=COLOR_ACCENT, width=3)))
                fig_b.add_trace(go.Scatter(x=df_bench['Date'], y=df_bench['SP500_Rebased'], name="S&P 500", line=dict(color="#64748b", width=2, dash="dot")))
                
                fig_b.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=350, legend=dict(orientation="h", y=1, x=0)
                )
                st.plotly_chart(fig_b, use_container_width=True)
        
        with c2:
            st.subheader("Drawdown (Profondeur de Baisse)")
            max_h = df_hist['Total'].cummax()
            dd = (df_hist['Total'] - max_h) / max_h * 100
            
            fig_dd = px.area(x=df_hist['Date'], y=dd, height=350)
            fig_dd.update_traces(line_color="#ef4444", fillcolor="rgba(239, 68, 68, 0.2)")
            fig_dd.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Drawdown %", xaxis_title=None
            )
            st.plotly_chart(fig_dd, use_container_width=True)
            
        # Matrice Mensuelle
        st.subheader("Matrice de Rentabilité Mensuelle")
        df_matrix = df_hist.copy()
        df_matrix['Year'] = df_matrix['Date'].dt.year
        df_matrix['Month'] = df_matrix['Date'].dt.month
        
        # Nettoyage Returns
        def clean_ret(x):
            if isinstance(x, str): return float(x.replace(',', '.').replace('%', ''))/100
            return float(x) if pd.notnull(x) else 0.0
            
        df_matrix['Daily_Return'] = df_matrix['PF_Return_TWR'].apply(clean_ret)
        m_rets = df_matrix.groupby(['Year', 'Month'])['Daily_Return'].apply(lambda x: (1+x).prod()-1).unstack(level=1).fillna(0) * 100
        
        fig_heat = px.imshow(m_rets, text_auto='.2f', color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect="auto")
        fig_heat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)

# --- ONGLET 3 : ALLOCATION & ACTIFS ---
with tab_alloc:
    col_sun, col_grid = st.columns([1, 2])
    
    with col_sun:
        st.subheader("Structure du Capital")
        if not df_pf.empty:
            # Sunburst Chart : Type -> Nom
            fig_sun = px.sunburst(
                df_pf, path=['Type', 'Nom'], values='Valo',
                color='Type', color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_sun.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=0, l=0, r=0, b=0), height=400
            )
            st.plotly_chart(fig_sun, use_container_width=True)
            
    with col_grid:
        st.subheader("Inventaire Détaillé")
        if not df_pf.empty:
            # Préparation d'un dataframe propre pour l'affichage
            df_display = df_pf[['Nom', 'Ticker', 'Type', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo', 'Perf_%', 'Poids_%']].copy()
            
            # Configuration des colonnes pour st.dataframe (Nouveauté Streamlit)
            st.dataframe(
                df_display,
                column_config={
                    "Nom": st.column_config.TextColumn("Actif", width="medium"),
                    "Quantité": st.column_config.NumberColumn("Qté", format="%.4f"),
                    "PRU": st.column_config.NumberColumn("PRU", format="%.2f €"),
                    "Prix_Actuel": st.column_config.NumberColumn("Prix", format="%.2f €"),
                    "Valo": st.column_config.NumberColumn("Valorisation", format="%.2f €"),
                    "Perf_%": st.column_config.ProgressColumn(
                        "Perf %", format="%.2f %%", min_value=-50, max_value=50,
                    ),
                    "Poids_%": st.column_config.NumberColumn("Poids", format="%.1f %%")
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )

    # Section "Top Movers" (Gagnants / Perdants)
    st.markdown("---")
    st.subheader("Top Movers (Séance & Global)")
    if not df_pf.empty:
        col_win, col_lose = st.columns(2)
        
        df_sorted = df_pf.sort_values(by="Perf_%", ascending=False)
        winner = df_sorted.iloc[0]
        loser = df_sorted.iloc[-1]
        
        with col_win:
            st.success(f"🏆 Meilleure Perf : **{winner['Nom']}**")
            st.metric("Performance", f"{winner['Perf_%']:+.2f} %", f"{winner['Valo']:,.2f} €")
            
        with col_lose:
            st.error(f"🥀 Plus fort repli : **{loser['Nom']}**")
            st.metric("Performance", f"{loser['Perf_%']:+.2f} %", f"{loser['Valo']:,.2f} €")

# FOOTER
st.markdown("---")
st.markdown("<center><small style='color: #64748b;'>ULTIMATE ESTATE V3.0 • SYSTÈME DE GESTION PRIVÉE</small></center>", unsafe_allow_html=True)