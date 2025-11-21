import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import io

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Ultimate Liquid Estate", layout="wide", page_icon="💎")

# --- 2. DESIGN SYSTEM "ULTIMATE GLASS" ---
st.markdown("""
<style>
    /* IMPORT POLICE MODERNE */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');

    /* FOND MAILLÉ (MESH GRADIENT) PLUS LUMINEUX */
    .stApp {
        background-color: #f0f4f8;
        background-image: 
            radial-gradient(at 0% 0%, hsla(210,100%,96%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(220,100%,93%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(190,100%,94%,1) 0, transparent 50%),
            radial-gradient(at 0% 100%, hsla(260,100%,95%,1) 0, transparent 50%),
            radial-gradient(at 80% 100%, hsla(240,100%,96%,1) 0, transparent 50%),
            radial-gradient(at 0% 50%, hsla(340,100%,96%,1) 0, transparent 50%);
        font-family: 'Outfit', sans-serif;
        color: #1e293b;
    }

    /* --- LE STYLE LIQUID GLASS AVANCÉ --- */
    
    /* Conteneurs génériques (Métriques, Charts) */
    div[data-testid="stMetric"], 
    div[data-testid="stDataFrame"], 
    div.stPlotlyChart, 
    div.stForm,
    div.stExpander {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.06),
            inset 0 0 0 1px rgba(255, 255, 255, 0.5); /* Double bordure interne */
        padding: 24px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }

    /* Effet de survol "Lévitation" */
    div[data-testid="stMetric"]:hover, div.stPlotlyChart:hover {
        transform: translateY(-6px);
        box-shadow: 
            0 20px 50px rgba(30, 41, 59, 0.12),
            inset 0 0 0 1px rgba(255, 255, 255, 0.9);
    }

    /* --- TYPOGRAPHIE PREMIUM --- */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Label des métriques (Petit texte) */
    div[data-testid="stMetricLabel"] {
        font-size: 15px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Valeur des métriques (Gros chiffres) */
    div[data-testid="stMetricValue"] {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 5px; /* Pour laisser place aux jambages */
    }

    /* --- ÉLÉMENTS SPÉCIFIQUES --- */

    /* Sidebar Givrée */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(30px);
        border-right: 1px solid rgba(255, 255, 255, 0.6);
    }

    /* Onglets "Pill" */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.4);
        padding: 8px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.5);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 12px;
        border: none;
        color: #64748b;
        font-weight: 600;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff;
        color: #0f172a;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* Boutons CTA */
    div.stButton > button {
        width: 100%;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.2);
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.3);
    }

    /* Custom CSS Class pour la "Hero Metric" (Total Patrimoine) */
    /* Cette classe sera appliquée artificiellement via l'ordre des éléments */
    
</style>
""", unsafe_allow_html=True)

# --- 3. DONNÉES & LOGIQUE (INCHANGÉES) ---

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

HIST_COLS = ["Date", "Total", "PEA", "BTC", "Plus-value", "Delta", "PV du Jour", "ESE", "Flux (€)", "PF_Return_TWR", "ESE_Return", "PF_Index100", "ESE_Index100"]

INITIAL_PORTFOLIO = {
    "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
    "Nom": ["BNP S&P 500", "Amundi World", "Lyxor Nasdaq", "Amundi USA x2", "Bitcoin", "Liquidités"],
    "Type": ["ETF Action", "ETF Action", "ETF Tech", "ETF Levier", "Crypto", "Cash"],
    "Quantité": [141.0, 716.0, 55.0, 176.0, 0.0142413, 510.84],
    "PRU": [24.41, 4.68, 71.73, 19.71, 90165.46, 1.00]
}

def safe_float(x):
    if pd.isna(x) or x == "": return 0.0
    s = str(x).strip().replace('"', '').replace('%', '').replace('€', '').replace(' ', '').replace(',', '.')
    try: return float(s)
    except: return 0.0

def load_state():
    if 'portfolio_df' not in st.session_state:
        if os.path.exists(FILE_PORTFOLIO):
            try:
                df = pd.read_csv(FILE_PORTFOLIO, sep=';')
                if df.shape[1] < 2: df = pd.read_csv(FILE_PORTFOLIO, sep=',')
            except: df = pd.DataFrame(INITIAL_PORTFOLIO)
        else:
            df = pd.DataFrame(INITIAL_PORTFOLIO)
            df.to_csv(FILE_PORTFOLIO, index=False, sep=';')
        df['Quantité'] = df['Quantité'].apply(safe_float)
        df['PRU'] = df['PRU'].apply(safe_float)
        st.session_state['portfolio_df'] = df

    if os.path.exists(FILE_HISTORY):
        try:
            with open(FILE_HISTORY, 'r') as f: raw_data = f.read()
            df_hist = pd.read_csv(io.StringIO(raw_data), sep=',', engine='python')
            for col in [c for c in df_hist.columns if c != "Date"]:
                df_hist[col] = df_hist[col].apply(safe_float)
            df_hist['Date'] = pd.to_datetime(df_hist['Date'], dayfirst=True, errors='coerce')
            df_hist = df_hist.dropna(subset=['Date'])
        except: df_hist = pd.DataFrame(columns=HIST_COLS)
    else: df_hist = pd.DataFrame(columns=HIST_COLS)
    return df_hist

def save_portfolio():
    st.session_state['portfolio_df'].to_csv(FILE_PORTFOLIO, index=False, sep=';')

def add_history_point(total, val_pea, val_btc, pv_totale, df_pf):
    df_hist = load_state()
    prev_total, prev_ese, prev_pf_idx, prev_ese_idx = (0.0, 0.0, 100.0, 100.0)
    
    if not df_hist.empty:
        last = df_hist.iloc[-1]
        prev_total = last.get('Total', 0)
        prev_ese = last.get('ESE', 0)
        prev_pf_idx = last.get('PF_Index100', 100) 
        if isinstance(prev_pf_idx, pd.Series): prev_pf_idx = prev_pf_idx.iloc[0]
        prev_ese_idx = last.get('ESE_Index100', 100)
        if isinstance(prev_ese_idx, pd.Series): prev_ese_idx = prev_ese_idx.iloc[0]
    else: prev_total = total

    try: ese_price = df_pf.loc[df_pf['Ticker'].str.contains("ESE"), 'Prix_Actuel'].values[0]
    except: ese_price = 0.0
    
    flux = 0.0
    delta = total - prev_total
    pv_jour = delta - flux
    denom = prev_total + flux
    pf_ret = (delta - flux) / denom if denom != 0 else 0.0
    ese_ret = (ese_price - prev_ese)/prev_ese if (prev_ese!=0 and ese_price!=0) else 0.0
    pf_idx = prev_pf_idx * (1 + pf_ret)
    ese_idx = prev_ese_idx * (1 + ese_ret)

    today_str = datetime.now().strftime("%d/%m/%Y")
    dates_str = []
    if not df_hist.empty: dates_str = df_hist['Date'].dt.strftime("%d/%m/%Y").values
    
    if today_str not in dates_str:
        new_row = {
            "Date": today_str, "Total": round(total, 2), "PEA": round(val_pea, 2), "BTC": round(val_btc, 2),
            "Plus-value": round(pv_totale, 2), "Delta": round(delta, 2), "PV du Jour": round(pv_jour, 2), 
            "ESE": round(ese_price, 2), "Flux (€)": 0, 
            "PF_Return_TWR": f"{pf_ret*100:.2f}%".replace('.', ','), "ESE_Return": f"{ese_ret*100:.2f}%".replace('.', ','),
            "PF_Index100": round(pf_idx, 2), "ESE_Index100": round(ese_idx, 2),
            "PF_Index100.1": round(pf_idx - 100, 2), "ESE_Index100.1": round(ese_idx - 100, 2)
        }
        if os.path.exists(FILE_HISTORY):
             line = ",".join([str(v) for v in new_row.values()])
             with open(FILE_HISTORY, 'a') as f: f.write("\n" + line)
        else: pd.DataFrame([new_row]).to_csv(FILE_HISTORY, index=False, sep=',')
        return True, delta
    return False, 0

df_history_static = load_state()

@st.cache_data(ttl=60)
def get_prices(tickers):
    prices = {"CASH": {"cur": 1.0, "prev": 1.0}}
    real = [t for t in tickers if t != "CASH"]
    if real:
        try:
            d = yf.download(real, period="5d", progress=False)['Close']
            if len(real) == 1:
                prices[real[0]] = {"cur": float(d.iloc[-1]), "prev": float(d.iloc[-2])}
            else:
                l, p = d.iloc[-1], d.iloc[-2]
                for t in real:
                    if t in l.index: prices[t] = {"cur": float(l[t]), "prev": float(p[t])}
        except: pass
    return prices

df = st.session_state['portfolio_df'].copy()
market = get_prices(df['Ticker'].unique())

df['Prix_Actuel'] = df['Ticker'].apply(lambda x: market.get(x, {}).get("cur", df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Prev'] = df['Ticker'].apply(lambda x: market.get(x, {}).get("prev", df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['PV'] = df['Valo'] - df['Investi']
df['Perf%'] = df.apply(lambda x: ((x['Prix_Actuel']-x['PRU'])/x['PRU']*100) if x['PRU']>0 else 0, axis=1)
df['Var_Jour'] = df['Valo'] - (df['Quantité'] * df['Prev'])

@st.cache_data(ttl=3600)
def get_market_monitor():
    # Liste exacte
    targets = {
        "INDEX CBOE:VIX": "^VIX",
        "CAC 40": "^FCHI",
        "SP 500": "^GSPC",
        "ETF MSCI World Amundi": "CW8.PA",
        "Emerging Market": "PAEEM.PA",
        "MSCI Europe": "PCEU.PA",
        "Euro Stoxx 50": "^STOXX50E",
        "Emerging Market ex Egypt": "PLEM.PA"
    }
    
    data = []
    
    try:
        tickers_list = list(targets.values())
        # On charge les données
        hist = yf.download(tickers_list, period="3mo", progress=False)['Close']
        
        for name, ticker in targets.items():
            if ticker in hist.columns:
                # Nettoyage et conversion explicite en nombres (float) pour éviter les erreurs de virgule
                series = hist[ticker].dropna().astype(float)
                
                if len(series) >= 22:
                    # Dernier prix connu (Live ou Clôture hier)
                    current = float(series.iloc[-1])
                    # Clôture précédente (pour perf jour)
                    prev = float(series.iloc[-2])
                    # Il y a exactement ~21 jours ouvrés de bourse dans 1 mois
                    month_ago = float(series.iloc[-21])
                    
                    # Calcul des variations (Mathématique pure)
                    # Résultat 0.05 donnera 5% à l'affichage
                    if prev != 0: perf_d = (current - prev) / prev
                    else: perf_d = 0.0
                        
                    if month_ago != 0: perf_m = (current - month_ago) / month_ago
                    else: perf_m = 0.0
                    
                    data.append({
                        "Indice": name,
                        "Ticker": ticker,
                        "Prix actuel": current,
                        "Perf. du Jour": perf_d,
                        "Perf 1 Mois": perf_m
                    })
            else:
                # Cas où la donnée est absente
                data.append({
                    "Indice": name, "Ticker": ticker, 
                    "Prix actuel": 0.0, "Perf. du Jour": 0.0, "Perf 1 Mois": 0.0
                })
                
    except Exception as e:
        st.error(f"Erreur données marché: {e}")
    
    # On retourne le DataFrame avec les colonnes dans l'ordre strict
    df_m = pd.DataFrame(data)
    return df_m[["Indice", "Ticker", "Prix actuel", "Perf. du Jour", "Perf 1 Mois"]]

# --- CALCULS TOTAUX ---
val_btc = df[df['Ticker'].str.contains("BTC")]['Valo'].sum()
val_pea = df[~df['Ticker'].str.contains("BTC")]['Valo'].sum()
total_pf = df['Valo'].sum()
total_pv = df['PV'].sum()

# --- MODIFICATION : CALCUL DE LA VARIATION (Total Actuel - Total Historique) ---
if not df_history_static.empty:
    # On récupère le dernier montant total enregistré dans le CSV (La veille)
    dernier_total_historique = df_history_static.iloc[-1]['Total']
    
    # Calcul : Aujourd'hui - Veille
    volat_jour_live = total_pf - dernier_total_historique
else:
    volat_jour_live = 0.0

def op_cash(amount):
    df = st.session_state['portfolio_df']
    mask = df['Ticker'] == 'CASH'
    if not mask.any():
         new_cash = pd.DataFrame([{"Ticker": "CASH", "Nom": "Liquidités", "Type": "Cash", "Quantité": 0.0, "PRU": 1.0}])
         df = pd.concat([df, new_cash], ignore_index=True)
         mask = df['Ticker'] == 'CASH'
    current = df.loc[mask, 'Quantité'].values[0]
    df.loc[mask, 'Quantité'] = current + amount
    st.session_state['portfolio_df'] = df
    save_portfolio()

def op_trade(sens, tick, q, p, nom=""):
    df = st.session_state['portfolio_df']
    if tick not in df['Ticker'].values:
        if sens=="Vente": return False, "Inconnu"
        new_row = pd.DataFrame([{"Ticker": tick, "Nom": nom, "Type": "Action", "Quantité": 0.0, "PRU": 0.0}])
        df = pd.concat([df, new_row], ignore_index=True)
    mask_a = df['Ticker'] == tick
    mask_c = df['Ticker'] == 'CASH'
    cash = df.loc[mask_c, 'Quantité'].values[0]
    cur_q = df.loc[mask_a, 'Quantité'].values[0]
    cur_p = df.loc[mask_a, 'PRU'].values[0]
    tot = q*p
    if sens=="Achat":
        if cash < tot: return False, "Manque Cash"
        new_q = cur_q + q
        new_p = ((cur_q*cur_p)+tot)/new_q
        df.loc[mask_a, 'Quantité'] = new_q
        df.loc[mask_a, 'PRU'] = new_p
        df.loc[mask_c, 'Quantité'] = cash - tot
    elif sens=="Vente":
        if cur_q < q: return False, "Manque Titres"
        df.loc[mask_a, 'Quantité'] = cur_q - q
        df.loc[mask_c, 'Quantité'] = cash + tot
    st.session_state['portfolio_df'] = df
    save_portfolio()
    return True, "Succès"

# --- 6. INTERFACE UTILISATEUR BENTO ---

# En-tête Minimaliste
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.markdown("## 🏛️ Patrimoine")
    st.caption(f"Dernière valorisation : {datetime.now().strftime('%d %B %Y • %H:%M')}")

# --- HERO SECTION (Le Gros Chiffre) ---
# On crée une "Box" spéciale pour le montant total
st.markdown("""
<div style="background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%); 
            padding: 30px; border-radius: 24px; border: 1px solid white; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 20px; text-align: center;">
    <p style="color: #64748b; font-size: 14px; text-transform: uppercase; letter-spacing: 2px; margin: 0;">Patrimoine Net</p>
    <h1 style="font-size: 56px; margin: 0; background: -webkit-linear-gradient(45deg, #0f172a, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        """ + f"{total_pf:,.2f} €" + """
    </h1>
    <p style="color: #10b981; font-weight: 600; margin-top: 5px;">
        """ + f"{volat_jour_live:+.2f} € (24h)" + """
    </p>
</div>
""", unsafe_allow_html=True)

# Calculs
MONTANT_INITIAL = 15450.00 
DATE_DEBUT = datetime(2022, 1, 1)
cash_dispo = df[df['Ticker']=='CASH']['Valo'].sum()
valo_investi = df[df['Ticker']!='CASH']['Valo'].sum()
cout_investi = df[df['Ticker']!='CASH']['Investi'].sum()
perf_totale_eur = total_pf - MONTANT_INITIAL
perf_totale_pct = (perf_totale_eur / MONTANT_INITIAL) * 100
perf_actif_eur = valo_investi - cout_investi
perf_actif_pct = (perf_actif_eur / cout_investi) * 100 if cout_investi != 0 else 0
days_held = (datetime.now() - DATE_DEBUT).days
years = days_held / 365.25
cagr = ((total_pf / MONTANT_INITIAL) ** (1/years) - 1) * 100 if years > 0 else 0
rendement_annuel = perf_totale_pct / years if years > 0 else 0

# --- GRID PRINCIPAL (BENTO) ---
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.markdown("##### 💰 Liquidité")
    st.metric("Cash Dispo", f"{cash_dispo:,.2f} €", delta=None)
    st.metric("Ratio Cash", f"{(cash_dispo/total_pf)*100:.1f} %")

with c2:
    st.markdown("##### 🚀 Performance")
    st.metric("Plus-Value Totale", f"{perf_totale_eur:+,.2f} €", f"{perf_totale_pct:+.2f} %")
    st.metric("Plus-Value Actifs", f"{perf_actif_eur:+,.2f} €", f"{perf_actif_pct:+.2f} %")

with c3:
    st.markdown("##### ⏳ Temps & Objectifs")
    st.metric("CAGR (Annuel)", f"{cagr:.2f} %")

st.markdown("---")

# --- SIDEBAR GESTION ---
with st.sidebar:
    st.header("🕹️ Centre de Contrôle")
    with st.expander("💰 Trésorerie", expanded=True):
        mnt = st.number_input("Montant (€)", step=100.0)
        if st.button("Valider Virement", type="secondary"):
            if mnt > 0: operation_tresorerie(mnt); st.success("OK"); st.rerun()
    st.write("")
    with st.expander("📈 Trading", expanded=True):
        sens = st.radio("Sens", ["Achat", "Vente"], horizontal=True)
        tickers = [t for t in df['Ticker'].unique() if t != "CASH"]
        mode = st.radio("Actif", ["Existant", "Nouveau"], horizontal=True, label_visibility="collapsed")
        if mode == "Existant":
            tick = st.selectbox("Sélection", tickers)
            nom = ""
        else:
            tick = st.text_input("Symbole (ex: AI.PA)").upper()
            nom = st.text_input("Nom")
        c_q, c_p = st.columns(2)
        qty = c_q.number_input("Qté", min_value=0.00000001, step=0.01, format="%.8f")
        price = c_p.number_input("Prix", min_value=0.01, step=0.01, format="%.2f")
        if st.button("Confirmer Ordre", type="primary"):
            ok, msg = op_trade(sens, tick, qty, price, nom)
            if ok: st.success(msg); st.rerun()
            else: st.error(msg)
    st.markdown("---")
    if st.button("💾 Sauvegarder Historique"):
        succes, d = add_history_point(total_pf, val_pea, val_btc, total_pv, df)
        if succes: st.success(f"Delta : {d:+.2f} €"); import time; time.sleep(1); st.rerun()
        else: st.warning("Déjà fait")

# --- ONGLETS & VISUALISATION ---
tab1, tab2, tab3, tab4 = st.tabs(["📋 Portefeuille", "📊 Graphiques", "🔮 Futur", "🔧 Admin"])

with tab1:
    st.dataframe(df[['Nom','Quantité','PRU','Prix_Actuel','Valo','Var_Jour','Perf%']], hide_index=True, use_container_width=True,
                 column_config={"Perf%": st.column_config.ProgressColumn(min_value=-30, max_value=30, format="%.2f %%"),
                                "Var_Jour": st.column_config.NumberColumn(format="%+.2f €"),
                                "Quantité": st.column_config.NumberColumn(format="%.8f"),
                                "Valo": st.column_config.NumberColumn(format="%.2f €")})

with tab2:
    if not df_history_static.empty:
        df_g = df_history_static.sort_values('Date').copy()
        
        col_g1, col_g2 = st.columns([2, 1])
        
        with col_g1:
            st.caption("Trajectoire Patrimoniale")
            fig_main = px.area(df_g, x='Date', y='Total')
            fig_main.update_layout(template="simple_white", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig_main.update_traces(line_color='#3b82f6', fillcolor='rgba(59, 130, 246, 0.1)')
            st.plotly_chart(fig_main, use_container_width=True)
            
        with col_g2:
            st.caption("Volatilité Quotidienne")
            if 'Delta' in df_g.columns:
                colors = ['#10b981' if v >= 0 else '#ef4444' for v in df_g['Delta']]
                fig_vol = go.Figure(go.Bar(x=df_g['Date'], y=df_g['Delta'], marker_color=colors))
                fig_vol.update_layout(template="simple_white", margin=dict(l=0,r=0,t=0,b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_vol, use_container_width=True)
                
        st.caption("Benchmark S&P 500")
        try:
            x = df_g['Date']
            col_idx = [c for c in df_g.columns if "Index100" in c]
            if len(col_idx) >= 2:
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=x, y=df_g[col_idx[0]], mode='lines', name='PF', line=dict(color='#0f172a', width=2)))
                fig_b.add_trace(go.Scatter(x=x, y=df_g[col_idx[1]], mode='lines', name='S&P', line=dict(color='#94a3b8', width=2, dash='dot')))
                fig_b.update_layout(template="simple_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_b, use_container_width=True)
        except: pass
    else: st.info("Pas d'historique.")
with tab2:
    # --- MONITOR DE MARCHÉ ---
    st.subheader("Données journalière et mensuelle")
    
    df_market = get_market_monitor()
    
    if not df_market.empty:
        st.dataframe(
            df_market,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Indice": st.column_config.TextColumn("Indice", width="medium"),
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Prix actuel": st.column_config.NumberColumn("Prix actuel", format="%.2f"),
                "Perf. du Jour": st.column_config.NumberColumn("Perf. du Jour", format="%.2f %%"),
                "Perf 1 Mois": st.column_config.NumberColumn("Perf 1 Mois", format="%.2f %%")
            }
        )
    
    st.markdown("---")
    
with tab3:
    c_in, c_out = st.columns([1,3])
    with c_in:
        add = st.number_input("Apport Mensuel", 500)
        r = st.slider("Taux %", 2.0, 12.0, 8.0)
        y = st.slider("Années", 5, 30, 15)
    with c_out:
        res = []
        c = total_pf
        for i in range(1, y+1):
            c = c*(1+r/100)+(add*12)
            res.append({"Année": datetime.now().year+i, "Capital": c})
        fig_p = px.area(pd.DataFrame(res), x="Année", y="Capital")
        fig_p.update_layout(template="simple_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_p.update_traces(line_color='#10b981', fillcolor='rgba(16, 185, 129, 0.1)')
        st.plotly_chart(fig_p, use_container_width=True)

with tab4:
    st.warning("Admin Zone")
    f_ch = st.radio("Fichier", ["Portefeuille", "Historique"], horizontal=True)
    if f_ch == "Portefeuille":
        ed = st.data_editor(df, num_rows="dynamic")
        if st.button("Save PF"): ed.to_csv(FILE_PORTFOLIO, index=False, sep=';'); st.success("OK"); st.rerun()
    else:
        if os.path.exists(FILE_HISTORY):
            with open(FILE_HISTORY, 'r') as f: raw = f.read()
            h_ed = st.data_editor(pd.read_csv(io.StringIO(raw), sep=',', engine='python'), num_rows="dynamic")
            if st.button("Save Hist"):
                if pd.api.types.is_datetime64_any_dtype(h_ed['Date']): h_ed['Date'] = h_ed['Date'].dt.strftime('%d/%m/%Y')
                h_ed.to_csv(FILE_HISTORY, index=False, sep=',', quotechar='"', quoting=1)
                st.success("OK"); st.rerun()