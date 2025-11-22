import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import textwrap

# Ajoutez ceci avec vos fonctions, par exemple après get_market_indices()

def create_bento_card(asset, card_bg, border_color, text_color, metric_gradient):
    # Calcul des couleurs dynamiques
    color_perf = "#10b981" if asset['Perf_%'] >= 0 else "#ef4444"
    bg_perf = "rgba(16, 185, 129, 0.15)" if asset['Perf_%'] >= 0 else "rgba(239, 68, 68, 0.15)"
    arrow = "▲" if asset['Perf_%'] >= 0 else "▼"
    
    # HTML compacté pour éviter les bugs d'indentation Markdown
    return f"""
    <div style="background-color: {card_bg}; border: 1px solid {border_color}; border-radius: 20px; padding: 20px; margin-bottom: 20px; transition: transform 0.2s;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-weight: 700; font-size: 1.1rem; color: {text_color}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 60%;">{asset['Nom']}</span>
            <span style="background-color: {bg_perf}; color: {color_perf}; padding: 4px 10px; border-radius: 10px; font-size: 0.85rem; font-weight: 600;">{arrow} {asset['Perf_%']:+.2f}%</span>
        </div>
        <div style="margin-bottom: 15px;">
            <div style="font-size: 0.85rem; opacity: 0.6; color: {text_color};">Valorisation</div>
            <div style="font-size: 1.8rem; font-weight: 800; background: {metric_gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; color: {text_color};">{asset['Valo']:,.2f} €</div>
        </div>
        <div style="display: flex; justify-content: space-between; border-top: 1px solid {border_color}; padding-top: 12px; font-size: 0.9rem; color: {text_color};">
            <div style="display: flex; flex-direction: column;"><span style="opacity: 0.5; font-size: 0.75rem;">Quantité</span><span style="font-weight: 500;">{asset['Quantité']:.4f}</span></div>
            <div style="display: flex; flex-direction: column; text-align: right;"><span style="opacity: 0.5; font-size: 0.75rem;">Prix Actuel</span><span style="font-weight: 500;">{asset['Prix_Actuel']:.2f} €</span></div>
        </div>
    </div>
    """

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tableau de Bord", layout="wide", page_icon="💎")

# --- 2. GESTION DU THÈME (DARK / LIGHT) ---
with st.sidebar:
    st.header("⚙️ Préférences")
    dark_mode = st.toggle("🌙 Mode Sombre", value=True)
    st.caption("Tableau de Bord V.1.6 (Production)")

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

# --- 3. FONCTIONS ROBUSTES ---

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# Chargement des données avec gestion d'erreur explicite
def load_data():
    # 1. Portefeuille
    if os.path.exists(FILE_PORTFOLIO):
        try:
            df_pf = pd.read_csv(FILE_PORTFOLIO, sep=',', dtype=str) # Le robot écrit avec des virgules
        except Exception as e:
            st.error(f"Erreur lecture portefeuille: {e}")
            df_pf = pd.DataFrame()
    else:
        df_pf = pd.DataFrame()

    # Nettoyage Portefeuille
    def clean_float(x):
        if pd.isna(x): return 0.0
        return float(str(x).replace(',', '.').replace('€', '').replace(' ', '').replace('%', ''))

    if not df_pf.empty:
        for c in ['Quantité', 'PRU']:
            if c in df_pf.columns: df_pf[c] = df_pf[c].apply(clean_float)

    # 2. Historique
    df_h = pd.DataFrame()
    if os.path.exists(FILE_HISTORY):
        try:
            # LE ROBOT ÉCRIT AVEC DES POINTS-VIRGULES (sep=';')
            df_h = pd.read_csv(FILE_HISTORY, sep=';', on_bad_lines='skip', engine='python')
            
            # Conversion Date
            df_h['Date'] = pd.to_datetime(df_h['Date'], dayfirst=True, errors='coerce')
            df_h = df_h.dropna(subset=['Date']).sort_values('Date')
            
            # Nettoyage des colonnes numériques pour les graphiques
            for col in ['Total', 'PF_Index100', 'ESE_Index100']:
                if col in df_h.columns:
                    df_h[col] = df_h[col].apply(clean_float)
                    
        except Exception as e:
            st.error(f"Erreur lecture historique: {e}")
            
    return df_pf, df_h

@st.cache_data(ttl=300)
def get_live_prices(tickers):
    prices = {"CASH": {"cur": 1.0, "prev": 1.0}}
    real_ticks = [t for t in tickers if t != "CASH" and isinstance(t, str)]
    if not real_ticks: return prices
    
    for t in real_ticks:
        try:
            hist = yf.Ticker(t).history(period="5d")
            if not hist.empty:
                cur = float(hist['Close'].iloc[-1])
                prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else cur
                prices[t] = {"cur": cur, "prev": prev}
            else:
                # Fallback
                data = yf.download(t, period="1d", progress=False)
                if not data.empty:
                    val = data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1]
                    prices[t] = {"cur": float(val), "prev": float(val)}
                else:
                    prices[t] = {"cur": 0.0, "prev": 0.0}
        except:
            prices[t] = {"cur": 0.0, "prev": 0.0}
    return prices

@st.cache_data(ttl=3600)
def get_market_indices():
    targets = {"S&P 500": "^GSPC", "CAC 40": "^FCHI", "Bitcoin": "BTC-EUR", "VIX": "^VIX"}
    res = []
    for name, tick in targets.items():
        try:
            h = yf.Ticker(tick).history(period="5d")
            if not h.empty:
                cur = float(h['Close'].iloc[-1])
                prev = float(h['Close'].iloc[-2]) if len(h)>1 else cur
                perf = ((cur-prev)/prev)*100 if prev != 0 else 0
                res.append({"Indice": name, "Prix": cur, "24h %": perf})
        except: pass
    return pd.DataFrame(res)

# --- 4. EXÉCUTION ---

df_pf, df_hist = load_data()

# Si le portefeuille est vide, on met des données par défaut pour ne pas planter
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
    # Sécurité prix 0
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

# --- 5. AFFICHAGE ---

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

# Calcul Volatilité (Annualisée)
# On reprend le nettoyage des pourcentages
def clean_pct_metric(x):
    try: return float(str(x).replace('%', '').replace(',', '.')) / 100
    except: return 0.0

if not df_hist.empty:
    daily_rets = df_hist['PF_Return_TWR'].apply(clean_pct_metric)
    volatility = daily_rets.std() * (252 ** 0.5) * 100 # Annualisation (252 jours de bourse)
else:
    volatility = 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Liquidité", f"{CASH_DISPO:,.2f} €", f"{(CASH_DISPO/TOTAL_ACTUEL)*100:.1f}% Alloc." if TOTAL_ACTUEL > 0 else "0%")
c2.metric("Plus-Value Latente", f"{PV_TOTALE:+,.2f} €", f"{(PV_TOTALE/(TOTAL_ACTUEL-PV_TOTALE))*100:.2f}%" if (TOTAL_ACTUEL-PV_TOTALE)!=0 else "0%")
c3.metric("CAGR (Annuel)", f"{cagr_val:.2f} %", f"Depuis {DATE_DEBUT.year}")
c4.metric("Volatilité", f"{volatility:.2f} %", "Risque Annualisé")

# --- GRAPHIQUES ---
st.markdown("---")
st.markdown("<div class='section-header'>📊 Analyse & Marché</div>", unsafe_allow_html=True)

if not df_hist.empty:
    # Calcul du Plus Haut Historique (All-Time High)
    max_histo = df_hist['Total'].max()
    
    # Le Drawdown est la distance actuelle par rapport à ce sommet
    # Formule : (Valeur Actuelle - Sommet) / Sommet
    drawdown = ((TOTAL_ACTUEL - max_histo) / max_histo) * 100
    
    # Affichage conditionnel élégant
    col_dd1, col_dd2 = st.columns([1, 3])
    
    with col_dd1:
        # On affiche le Drawdown en rouge s'il est significatif, sinon en vert (proche du sommet)
        color_dd = "off" if drawdown > -1 else "inverse" 
        st.metric("Drawdown (Depuis Sommet)", f"{drawdown:.2f} %", delta_color="off")
        st.caption(f"Plus Haut Historique : {max_histo:,.0f} €")
    
    with col_dd2:
        if drawdown > -5:
             st.info("💎 **Solidité :** Votre portefeuille est proche de son sommet historique.")
        elif drawdown > -15:
             st.warning("⚠️ **Correction :** Le marché respire. Opportunité de renforcement ?")
        else:
             st.error("🚨 **Bear Market :** Zone d'achat agressive pour le long terme.")

if not df_hist.empty and len(df_hist) > 1:
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Trajectoire Patrimoniale")
        # Graphique Aire
        fig = px.area(df_hist, x='Date', y='Total', line_shape='spline')
        fig.update_layout(
            template="plotly_dark" if dark_mode else "simple_white",
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
    if df_hist.empty:
        st.error("❌ Fichier historique introuvable ou illisible.")
        st.info("Vérifiez que vous avez bien fait 'git push' après avoir lancé le robot.")
    else:
        st.warning("⚠️ Historique insuffisant (1 seule date). Attendez demain ou lancez master_reset.py.")

# --- MARCHÉS ---
st.caption("Pulsation Mondiale")
df_m = get_market_indices()
if not df_m.empty:
    cols = st.columns(len(df_m))
    for i, row in df_m.iterrows():
        with cols[i]:
            st.metric(row['Indice'], f"{row['Prix']:.2f}", f"{row['24h %']:+.2f}%")
# ... après le calcul des dataframes ...

# ... (Code précédent inchangé)

# Création du Donut Chart
fig_alloc = px.pie(
    df_pf, 
    values='Valo', 
    names='Nom', 
    title='Répartition des Actifs',
    hole=0.6, 
    color_discrete_sequence=px.colors.qualitative.Pastel if not dark_mode else px.colors.qualitative.Bold
)

# ... (Création de fig_alloc avec px.pie inchangée) ...

fig_alloc.update_layout(
    template="plotly_dark" if dark_mode else "plotly_white",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=True,
    # 1. On ancre la légende
    legend=dict(
        orientation="v", 
        y=0.5, 
        x=1.02, # Juste un tout petit peu décalé du cercle
        xanchor="left", 
        yanchor="middle"
    ),
    # 2. LA CLÉ EST ICI : On force une marge à droite (r=160)
    margin=dict(t=30, b=30, l=20, r=200) 
)

st.plotly_chart(fig_alloc, use_container_width=True)

# ... (Reste du code inchangé)

# --- DETAIL ---
# ... (Code précédent inchangé) ...

# ... (Le code précédent reste inchangé)

# ... (tout votre code précédent reste inchangé) ...

# --- DETAIL (STYLE BENTO) ---
st.markdown("---")
st.markdown("<div class='section-header'>📋 Détail des Actifs</div>", unsafe_allow_html=True)

if not df_pf.empty:
    # Configuration de la grille (3 cartes par ligne)
    COLS = 3
    rows = [df_pf.iloc[i:i + COLS] for i in range(0, len(df_pf), COLS)]

    for row_data in rows:
        cols = st.columns(COLS)
        for i, (index, asset) in enumerate(row_data.iterrows()):
            with cols[i]:
                # Appel de la fonction propre
                html_card = create_bento_card(asset, card_bg, border_color, text_color, metric_gradient)
                # Affichage (le .strip() supprime les derniers espaces gênants)
                st.markdown(html_card.strip(), unsafe_allow_html=True)
else:
    st.info("Aucun actif à afficher.")

# --- MATRICE DE PERFORMANCE (HEDGE FUND STYLE) ---
if not df_hist.empty:
    st.markdown("<div class='section-header'>📅 Performance Mensuelle</div>", unsafe_allow_html=True)
    
    # Préparation des données
    df_matrix = df_hist.copy()
    df_matrix['Year'] = df_matrix['Date'].dt.year
    df_matrix['Month'] = df_matrix['Date'].dt.month
    # Calcul du rendement mensuel : (Fin du mois / Début du mois) - 1
    # Simplification : on prend le rendement journalier cumulé
    
    # On convertit la colonne 'PF_Return_TWR' (qui est en string 'x,xx%') en float
    def clean_pct(x):
        if isinstance(x, str):
            return float(x.replace('%', '').replace(',', '.')) / 100
        return 0.0
        
    df_matrix['Daily_Return'] = df_matrix['PF_Return_TWR'].apply(clean_pct)
    
    # Pivot Table par Année/Mois
    # Note: C'est une approximation, pour être précis il faudrait le NAV de fin de mois vs début de mois
    # Calcul "Compound" pour une précision exacte
    monthly_returns = df_matrix.groupby(['Year', 'Month'])['Daily_Return'].apply(lambda x: (1 + x).prod() - 1) * 100 
    
    matrix = monthly_returns.unstack(level=1).fillna(0)
    
    # Renommer les colonnes mois (1->Jan, etc.)
    months_map = {1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Juin', 
                  7: 'Juil', 8: 'Août', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'}
    matrix.columns = [months_map.get(c, c) for c in matrix.columns]
    
    # Affichage Heatmap via Plotly
    fig_heat = px.imshow(
        matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale="RdBu", # Rouge à Bleu (ou Red-Green si custom)
        color_continuous_midpoint=0
    )
    fig_heat.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- PROJECTION PATRIMONIALE ---
st.markdown("---")
st.markdown("<div class='section-header'>🔮 Projection : Liberté Financière</div>", unsafe_allow_html=True)

# Paramètres de simulation dans un Expander pour ne pas polluer
with st.expander("⚙️ Paramètres de Simulation", expanded=True):
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    epargne_mensuelle = col_sim1.number_input("Épargne Mensuelle (€)", value=1000, step=100)
    annees_proj = col_sim2.slider("Horizon (Années)", 5, 30, 15)
    rendement_hypoth = col_sim3.slider("Hypothèse Rendement (%)", 2.0, 15.0, float(cagr_val) if cagr_val > 0 else 8.0)

# Calcul de la projection
future_data = []
capital = TOTAL_ACTUEL
dates_future = []
current_year = datetime.now().year

for i in range(1, (annees_proj * 12) + 1):
    # Apport mensuel + Rendement mensuel composé
    r_mensuel = (1 + rendement_hypoth/100)**(1/12) - 1
    capital = (capital + epargne_mensuelle) * (1 + r_mensuel)
    
    # On ajoute un point de donnée par an pour alléger le graph
    if i % 12 == 0:
        future_data.append(round(capital, 2))
        dates_future.append(current_year + (i // 12))

# Création du DataFrame de projection
df_proj = pd.DataFrame({"Année": dates_future, "Patrimoine Projeté": future_data})

# Graphique de Projection
fig_proj = px.bar(df_proj, x="Année", y="Patrimoine Projeté", text="Patrimoine Projeté")
fig_proj.update_traces(
    marker_color=chart_line_color, 
    texttemplate='%{text:.2s} €', 
    textposition='outside'
)
fig_proj.update_layout(
    template="plotly_dark" if dark_mode else "plotly_white",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=400,
    margin=dict(t=30, l=0, r=0, b=0),
    yaxis=dict(showgrid=True, gridcolor=border_color)
)

st.plotly_chart(fig_proj, use_container_width=True)

# Phrase de conclusion dynamique
capital_final = future_data[-1] if future_data else 0
rente_mensuelle = (capital_final * 0.04) / 12 # Règle des 4% de retrait safe
st.info(f"🚀 Avec **{epargne_mensuelle}€** par mois et un rendement de **{rendement_hypoth}%**, vous auriez **{capital_final:,.0f} €** dans {annees_proj} ans. Cela génèrerait une rente passive estimée (4%) de **{rente_mensuelle:,.0f} € / mois**.")