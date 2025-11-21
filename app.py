import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import io

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Liquid Mesh Estate", layout="wide", page_icon="💎")

# --- 2. DESIGN SYSTEM "LIQUID MESH" ---
st.markdown("""
<style>
    /* IMPORT POLICE */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700;800&display=swap');

    /* FOND D'ÉCRAN MESH GRADIENT */
    .stApp {
        background-color: #f8fafc;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,90%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,90%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,90%,1) 0, transparent 50%),
            radial-gradient(at 0% 100%, hsla(280,100%,93%,1) 0, transparent 50%),
            radial-gradient(at 100% 100%, hsla(180,100%,90%,1) 0, transparent 50%);
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    /* LE STYLE LIQUID GLASS (Standard Streamlit Widgets) */
    div[data-testid="stMetric"], 
    div[data-testid="stDataFrame"], 
    div.stPlotlyChart, 
    div.stForm,
    div.stExpander {
        background-color: rgba(255, 255, 255, 0.45) !important; 
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.07), 
            inset 2px 2px 1px 0 rgba(255, 255, 255, 0.7), 
            inset -1px -1px 1px 1px rgba(255, 255, 255, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.8);
        padding: 20px !important;
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover, div.stPlotlyChart:hover {
        transform: translateY(-5px);
        background-color: rgba(255, 255, 255, 0.55) !important;
        box-shadow: 
            0 15px 35px rgba(31, 38, 135, 0.1),
            inset 2px 2px 1px 0 rgba(255, 255, 255, 0.9), 
            inset -1px -1px 1px 1px rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(255, 255, 255, 1);
    }

    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.02em;
        text-shadow: 0 1px 2px rgba(255,255,255,0.5);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 30px;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #0f172a, #334155);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div[data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 600;
        font-size: 14px;
    }

    /* CUSTOM CSS POUR LA JAUGE LIQUIDE (Harmonisé Bento) */
    .liquid-card {
        /* Exactement les mêmes propriétés que les stMetric */
        background-color: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.07), 
            inset 2px 2px 1px 0 rgba(255, 255, 255, 0.7), 
            inset -1px -1px 1px 1px rgba(255, 255, 255, 0.4);
        padding: 20px;
        margin-top: 15px; /* Espacement vertical standard */
        transition: all 0.3s ease;
    }
    
    .liquid-card:hover {
        transform: translateY(-5px);
        background-color: rgba(255, 255, 255, 0.55);
        box-shadow: 
            0 15px 35px rgba(31, 38, 135, 0.1),
            inset 2px 2px 1px 0 rgba(255, 255, 255, 0.9), 
            inset -1px -1px 1px 1px rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(255, 255, 255, 1);
    }
    
    .progress-container {
        background: rgba(255,255,255,0.5);
        border-radius: 10px;
        height: 12px;
        width: 100%;
        margin-top: 12px;
        box-shadow: inset 1px 1px 3px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.6);
        overflow: hidden;
    }
    
    .progress-bar-liquid {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%);
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.4);
        transition: width 1s ease-in-out;
    }

    .goal-label {
        color: #475569;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 2px;
    }
    
    .goal-value {
        font-size: 30px;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #0f172a, #334155);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .goal-percent {
        float: right;
        color: #10b981;
        font-size: 14px;
        font-weight: 600;
        margin-top: 8px;
    }

    /* SIDEBAR & TABS */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.35);
        backdrop-filter: blur(25px);
        border-right: 1px solid rgba(255, 255, 255, 0.7);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px; background-color: rgba(255,255,255,0.3); padding: 10px; border-radius: 20px; margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.45); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.5);
        color: #475569; font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%); color: #0f172a; border: 1px solid #ffffff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    div.stButton > button {
        border-radius: 14px; background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white; border: none; font-weight: 600; padding: 0.5rem 1.2rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.2); transition: all 0.3s;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(15, 23, 42, 0.3); }

</style>
""", unsafe_allow_html=True)

# --- 3. CONSTANTES & FICHIERS ---

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

HIST_COLS = [
    "Date", "Total", "PEA", "BTC", "Plus-value", "Delta", "PV du Jour", 
    "ESE", "Flux (€)", "PF_Return_TWR", "ESE_Return", 
    "PF_Index100", "ESE_Index100"
]

INITIAL_PORTFOLIO = {
    "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
    "Nom": ["BNP S&P 500", "Amundi World", "Lyxor Nasdaq", "Amundi USA x2", "Bitcoin", "Liquidités"],
    "Type": ["ETF Action", "ETF Action", "ETF Tech", "ETF Levier", "Crypto", "Cash"],
    "Quantité": [141.0, 716.0, 55.0, 176.0, 0.01275433, 510.84], 
    "PRU": [24.41, 4.68, 71.73, 19.71, 90165.46, 1.00]
}

# --- 4. FONCTIONS ---

def safe_float(x):
    if pd.isna(x) or x == "": return 0.0
    s = str(x).strip()
    s = s.replace('"', '').replace('%', '').replace('€', '').replace(' ', '')
    s = s.replace(',', '.')
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
            cols_to_clean = [c for c in df_hist.columns if c != "Date"]
            for col in cols_to_clean:
                df_hist[col] = df_hist[col].apply(safe_float)
            df_hist['Date'] = pd.to_datetime(df_hist['Date'], dayfirst=True, errors='coerce')
            df_hist = df_hist.dropna(subset=['Date'])
        except:
            df_hist = pd.DataFrame(columns=HIST_COLS)
    else:
        df_hist = pd.DataFrame(columns=HIST_COLS)
    return df_hist

def save_portfolio():
    st.session_state['portfolio_df'].to_csv(FILE_PORTFOLIO, index=False, sep=';')

def add_history_point(total, val_pea, val_btc, pv_totale, df_pf):
    df_hist = load_state()
    
    # Récupération Veille
    prev_total = 0.0
    prev_ese = 0.0
    prev_pf_idx = 100.0
    prev_ese_idx = 100.0
    
    if not df_hist.empty:
        last = df_hist.iloc[-1]
        prev_total = last.get('Total', 0)
        prev_ese = last.get('ESE', 0)
        prev_pf_idx = last.get('PF_Index100', 100) 
        if isinstance(prev_pf_idx, pd.Series): prev_pf_idx = prev_pf_idx.iloc[0]
        prev_ese_idx = last.get('ESE_Index100', 100)
        if isinstance(prev_ese_idx, pd.Series): prev_ese_idx = prev_ese_idx.iloc[0]
    else:
        prev_total = total

    try: ese_price = df_pf.loc[df_pf['Ticker'].str.contains("ESE"), 'Prix_Actuel'].values[0]
    except: ese_price = 0.0
    
    flux = 0.0
    delta = total - prev_total # Calcul Delta Strict
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
            "Date": today_str,
            "Total": round(total, 2), 
            "PEA": round(val_pea, 2), 
            "BTC": round(val_btc, 2),
            "Plus-value": round(pv_totale, 2), 
            "Delta": round(delta, 2), 
            "PV du Jour": round(pv_jour, 2), 
            "ESE": round(ese_price, 2), 
            "Flux (€)": 0, 
            "PF_Return_TWR": f"{pf_ret*100:.2f}%".replace('.', ','),
            "ESE_Return": f"{ese_ret*100:.2f}%".replace('.', ','),
            "PF_Index100": round(pf_idx, 2), 
            "ESE_Index100": round(ese_idx, 2),
            "PF_Index100.1": round(pf_idx - 100, 2),
            "ESE_Index100.1": round(ese_idx - 100, 2)
        }
        if os.path.exists(FILE_HISTORY):
             line = ",".join([str(v) for v in new_row.values()])
             with open(FILE_HISTORY, 'a') as f: f.write("\n" + line)
        else:
             pd.DataFrame([new_row]).to_csv(FILE_HISTORY, index=False, sep=',')
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

# --- CALCULS TOTAUX ---
val_btc = df[df['Ticker'].str.contains("BTC")]['Valo'].sum()
val_pea = df[~df['Ticker'].str.contains("BTC")]['Valo'].sum()
total_pf = df['Valo'].sum()
total_pv = df['PV'].sum()

# --- CALCUL VARIATION JOUR COMPTABLE (Total Actuel - Total Historique J-1) ---
if not df_history_static.empty:
    dernier_total_historique = df_history_static.iloc[-1]['Total']
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
        if cash < tot: return False, "Cash manquant"
        new_q = cur_q + q
        new_p = ((cur_q*cur_p)+tot)/new_q
        df.loc[mask_a, 'Quantité'] = new_q
        df.loc[mask_a, 'PRU'] = new_p
        df.loc[mask_c, 'Quantité'] = cash - tot
    elif sens=="Vente":
        if cur_q < q: return False, "Qté insuffisante"
        df.loc[mask_a, 'Quantité'] = cur_q - q
        df.loc[mask_c, 'Quantité'] = cash + tot
    st.session_state['portfolio_df'] = df
    save_portfolio()
    return True, "Succès"

# --- 6. INTERFACE ---

st.markdown("<h1 style='text-align: center; margin-bottom: 30px; font-weight: 300;'>LIQUID MESH <span style='font-weight:700'>ESTATE</span></h1>", unsafe_allow_html=True)

# KPI
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

# BENTO ROW 1
st.markdown("#### 🔭 Vue Satellite")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portefeuille Net", f"{total_pf:,.2f} €")
c2.metric("Montant Initial", f"{MONTANT_INITIAL:,.2f} €")
c3.metric("Trésorerie", f"{cash_dispo:,.2f} €")
c4.metric("Variation Jour", f"{volat_jour_live:+,.2f} €", help="Vs dernière sauvegarde")

st.write("")

# BENTO ROW 2
st.markdown("#### 🚀 Performance Actifs")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Investi", f"{valo_investi:,.2f} €")
c6.metric("Coût (PRU)", f"{cout_investi:,.2f} €")
c7.metric("Perf. Actifs", f"{perf_actif_eur:+,.2f} €", f"{perf_actif_pct:+.2f} %")
c8.metric("Perf. Totale", f"{perf_totale_eur:+,.2f} €", f"{perf_totale_pct:+.2f} %")

st.write("")

# BENTO ROW 3 (AVEC JAUGE HARMONISÉE)
col_kpi_time, col_graph_time = st.columns([1, 3])

with col_kpi_time:
    st.markdown("#### ⏳ Temps")
    st.metric("Rendement/An", f"{rendement_annuel:.2f} %")
    st.metric("CAGR", f"{cagr:.2f} %")
    
    # --- JAUGE INTÉGRÉE EN STYLE NATIVE ---
    TARGET_GOAL = 100000.0 
    percent_goal = min((total_pf / TARGET_GOAL) * 100, 100.0)
    
    st.markdown(f"""
    <div class="liquid-card" style="margin-top: 15px; padding: 24px;">
        <div class="goal-label">Objectif 100k€</div>
        <div class="goal-value">{total_pf:,.0f} €</div>
        <div class="goal-percent">Avancement : {int(percent_goal)}%</div>
        <div class="progress-container">
            <div class="progress-bar-liquid" style="width: {percent_goal}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_graph_time:
    if not df_history_static.empty:
        df_g = df_history_static.sort_values('Date').copy()
        fig_mini = px.area(df_g, x='Date', y='Total', height=320) # Hauteur ajustée
        fig_mini.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, title=None), 
            yaxis=dict(showgrid=False, title=None, visible=False),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        fig_mini.update_traces(line_color='#0f172a', fillcolor='rgba(15, 23, 42, 0.05)')
        st.plotly_chart(fig_mini, use_container_width=True)

st.markdown("---")

# SIDEBAR
with st.sidebar:
    st.header("🕹️ Centre de Contrôle")
    with st.expander("💰 Trésorerie", expanded=True):
        mnt = st.number_input("Montant (€)", step=100.0)
        if st.button("Valider Virement", type="secondary", use_container_width=True):
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
        c1, c2 = st.columns(2)
        qty = c1.number_input("Qté", min_value=0.00000001, step=0.01, format="%.8f")
        price = c2.number_input("Prix", min_value=0.01, step=0.01, format="%.2f")
        st.caption(f"Total: {qty*price:,.2f}€")
        if st.button("Confirmer", type="primary", use_container_width=True):
            ok, msg = op_trade(sens, tick, qty, price, nom)
            if ok: st.success(msg); st.rerun()
            else: st.error(msg)
    st.markdown("---")
    if st.button("💾 Sauvegarder Historique"):
        succes, d = add_history_point(total_pf, val_pea, val_btc, total_pv, df)
        if succes: st.success(f"Sauvegardé ! Delta : {d:+.2f} €"); import time; time.sleep(1); st.rerun()
        else: st.warning("Déjà fait aujourd'hui")

# ONGLETS
tab1, tab2, tab3, tab4 = st.tabs(["📋 Positions", "📊 Benchmarks", "🔮 Projection", "🔧 Admin"])

with tab1:
    st.dataframe(df[['Nom','Quantité','PRU','Prix_Actuel','Valo','Var_Jour','Perf%']], hide_index=True, use_container_width=True,
                 column_config={"Perf%": st.column_config.ProgressColumn(min_value=-30, max_value=30, format="%.2f %%"),
                                "Var_Jour": st.column_config.NumberColumn(format="%+.2f €"),
                                "Quantité": st.column_config.NumberColumn(format="%.8f"),
                                "Valo": st.column_config.NumberColumn(format="%.2f €")})

with tab2:
    if not df_history_static.empty:
        df_g = df_history_static.sort_values('Date').copy()
        st.subheader("Performance vs S&P 500 (Base 100)")
        try:
            x_axis = df_g['Date']
            idx_cols = [c for c in df_g.columns if "Index100" in c]
            if len(idx_cols) >= 2:
                y_pf = df_g[idx_cols[0]]
                y_ese = df_g[idx_cols[1]]
                fig_bench = go.Figure()
                fig_bench.add_trace(go.Scatter(x=x_axis, y=y_pf, mode='lines', name='Portefeuille', line=dict(color='#0f172a', width=3)))
                fig_bench.add_trace(go.Scatter(x=x_axis, y=y_ese, mode='lines', name='S&P 500', line=dict(color='#94a3b8', width=2, dash='dot')))
                fig_bench.update_layout(template="simple_white", hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_bench, use_container_width=True)
        except: st.error("Données indices manquantes")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Volatilité")
            if 'Delta' in df_g.columns:
                colors = ['#10b981' if v >= 0 else '#ef4444' for v in df_g['Delta']]
                fig_vol = go.Figure(go.Bar(x=df_g['Date'], y=df_g['Delta'], marker_color=colors))
                fig_vol.update_layout(template="simple_white", margin=dict(l=0,r=0,t=0,b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_vol, use_container_width=True)
        with c2:
            st.subheader("Plus-Value Cumulée")
            fig_pv = px.area(df_g, x='Date', y='Plus-value')
            fig_pv.update_traces(line_color='#10b981', fillcolor='rgba(16, 185, 129, 0.1)')
            fig_pv.update_layout(template="simple_white", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pv, use_container_width=True)
    else: st.info("Historique vide.")

with tab3:
    st.subheader("Futur")
    col1, col2 = st.columns([1,2])
    with col1:
        add = st.number_input("Apport/mois", 500)
        r = st.slider("Rendement %", 2.0, 12.0, 8.0)
        y = st.slider("Années", 5, 30, 15)
    with col2:
        res = []
        c = total_pf
        for i in range(1, y+1):
            c = c*(1+r/100) + (add*12)
            res.append({"Année": datetime.now().year+i, "Capital": c})
        fig_proj = px.area(pd.DataFrame(res), x="Année", y="Capital")
        fig_proj.update_layout(template="simple_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_proj.update_traces(line_color='#0f172a', fillcolor='rgba(15, 23, 42, 0.1)')
        st.plotly_chart(fig_proj, use_container_width=True)

with tab4:
    st.warning("Zone Admin.")
    file_choice = st.radio("Fichier", ["Portefeuille", "Historique"], horizontal=True)
    if file_choice == "Portefeuille":
        edited_df = st.data_editor(df, num_rows="dynamic")
        if st.button("💾 Sauvegarder Portefeuille"):
            edited_df.to_csv(FILE_PORTFOLIO, index=False, sep=';')
            st.success("OK"); st.rerun()
    else:
        if os.path.exists(FILE_HISTORY):
            try:
                with open(FILE_HISTORY, 'r') as f: raw = f.read()
                df_h_edit = pd.read_csv(io.StringIO(raw), sep=',', engine='python')
                edited_hist = st.data_editor(df_h_edit, num_rows="dynamic")
                if st.button("💾 Sauvegarder Historique"):
                    if pd.api.types.is_datetime64_any_dtype(edited_hist['Date']):
                        edited_hist['Date'] = edited_hist['Date'].dt.strftime('%d/%m/%Y')
                    edited_hist.to_csv(FILE_HISTORY, index=False, sep=',', quotechar='"', quoting=1)
                    st.success("OK"); st.rerun()
            except Exception as e: st.error(f"Erreur: {e}")