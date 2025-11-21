import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import io

# --- 1. CONFIGURATION & DESIGN SYSTEM "LIQUID BENTO" ---
st.set_page_config(page_title="Gestion Patrimoniale Premium", layout="wide", page_icon="💎")

# CSS AVANCÉ : GLASSMORPHISM & BENTO GRID
st.markdown("""
<style>
    /* FOND GLOBAL : Dégradé subtil et maillé pour effet de profondeur */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(242, 246, 252) 0%, rgb(224, 233, 245) 90%);
        font-family: 'Inter', sans-serif;
    }

    /* TITRES */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* EFFET LIQUID GLASS (Le cœur du design) */
    /* Appliqué aux Métriques, Dataframes, et Graphiques */
    div[data-testid="stMetric"], div[data-testid="stDataFrame"], div.stPlotlyChart, div.stForm {
        background: rgba(255, 255, 255, 0.65) !important; /* Blanc semi-transparent */
        backdrop-filter: blur(16px); /* Flou d'arrière-plan (Effet Verre) */
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.8); /* Bordure subtile blanche */
        border-radius: 24px; /* Arrondis "Bento" prononcés */
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05); /* Ombre douce colorée */
        padding: 20px !important;
        transition: transform 0.2s ease;
    }

    /* Effet de survol sur les cartes */
    div[data-testid="stMetric"]:hover, div.stPlotlyChart:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 1);
    }

    /* ONGLETS MODERNISÉS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        color: #64748b;
        border: 1px solid rgba(255, 255, 255, 0.5);
        font-weight: 600;
        backdrop-filter: blur(4px);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0f172a;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.3);
    }

    /* Sidebar plus propre */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Couleurs spécifiques pour les valeurs positives/négatives dans les métriques */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONSTANTES & FICHIERS ---

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

# --- 3. FONCTIONS DE LECTURE INTELLIGENTE ---

def safe_float(x):
    if pd.isna(x) or x == "": return 0.0
    s = str(x).strip()
    s = s.replace('"', '').replace('%', '').replace('€', '').replace(' ', '')
    s = s.replace(',', '.')
    try: return float(s)
    except: return 0.0

def load_state():
    # 1. Portefeuille
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

    # 2. Historique
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
        prev_ese = 0.0
        prev_pf_idx = 100.0
        prev_ese_idx = 100.0

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

# --- 4. PRIX LIVE ---

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

val_btc = df[df['Ticker'].str.contains("BTC")]['Valo'].sum()
val_pea = df[~df['Ticker'].str.contains("BTC")]['Valo'].sum()
total_pf = df['Valo'].sum()
total_pv = df['PV'].sum()
volat_jour_live = df['Var_Jour'].sum()

# --- 5. OPÉRATIONS ---
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

# --- 6. INTERFACE & BENTO GRID ---

st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🏛️ Wealth Dashboard</h1>", unsafe_allow_html=True)

# Calculs KPI Avancés
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

# --- BENTO ROW 1 : Synthèse (4 blocs larges) ---
st.markdown("### 🔭 Vue Satellite")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portefeuille Net", f"{total_pf:,.2f} €", help="Valeur Totale")
c2.metric("Montant Initial", f"{MONTANT_INITIAL:,.2f} €", delta=None)
c3.metric("Trésorerie (Cash)", f"{cash_dispo:,.2f} €", f"{(cash_dispo/total_pf)*100:.1f}% du PF")
c4.metric("Variation du Jour", f"{volat_jour_live:+,.2f} €", help="P&L depuis hier clôture")

st.write("") # Espacement

# --- BENTO ROW 2 : Performance (4 blocs) ---
st.markdown("### 🚀 Performance")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Investi (Marché)", f"{valo_investi:,.2f} €")
c6.metric("Coût d'achat (PRU)", f"{cout_investi:,.2f} €")
c7.metric("Perf. Actifs", f"{perf_actif_eur:+,.2f} €", f"{perf_actif_pct:+.2f} %")
c8.metric("Perf. Totale", f"{perf_totale_eur:+,.2f} €", f"{perf_totale_pct:+.2f} %")

st.write("")

# --- BENTO ROW 3 : Temps (2 blocs larges + graph) ---
col_kpi_time, col_graph_time = st.columns([1, 3])
with col_kpi_time:
    st.markdown("### ⏳ Temps")
    st.metric("Rendement Annuel", f"{rendement_annuel:.2f} %")
    st.metric("CAGR (Composé)", f"{cagr:.2f} %")

with col_graph_time:
    # Graphique intégré dans le Bento
    if not df_history_static.empty:
        df_g = df_history_static.sort_values('Date').copy()
        fig_mini = px.area(df_g, x='Date', y='Total', height=250)
        fig_mini.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title=None, yaxis_title=None,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        fig_mini.update_traces(line_color='#0f172a', fillcolor='rgba(15, 23, 42, 0.1)')
        st.plotly_chart(fig_mini, use_container_width=True)

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🕹️ Centre de Contrôle")
    
    with st.expander("💰 Trésorerie (Apport)", expanded=True):
        st.caption("Ajouter du Cash")
        mnt = st.number_input("Montant (€)", step=100.0)
        if st.button("Valider Virement", type="secondary", use_container_width=True):
            if mnt > 0:
                operation_tresorerie(mnt)
                st.success("Effectué !")
                st.rerun()

    st.write("")

    with st.expander("📈 Trading (Ordres)", expanded=True):
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
        
        st.caption(f"Total Estimé : {qty*price:,.2f}€")
        
        if st.button("Confirmer Ordre", type="primary", use_container_width=True):
            ok, msg = operation_trading(sens, tick, qty, price, nom)
            if ok: st.success(msg); st.rerun()
            else: st.error(msg)
    
    st.markdown("---")
    if st.button("💾 Sauvegarder Historique"):
        succes, d = add_history_point(total_pf, val_pea, val_btc, total_pv, df)
        if succes: 
            st.success(f"Sauvegardé ! Delta : {d:+.2f} €")
            import time; time.sleep(1); st.rerun()
        else: 
            st.warning("Déjà fait aujourd'hui")

# --- ONGLETS DÉTAILLÉS ---
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
                fig_bench.add_trace(go.Scatter(x=x_axis, y=y_pf, mode='lines', name='Mon Portefeuille', line=dict(color='#0f172a', width=3)))
                fig_bench.add_trace(go.Scatter(x=x_axis, y=y_ese, mode='lines', name='S&P 500', line=dict(color='#94a3b8', width=2, dash='dot')))
                fig_bench.update_layout(template="simple_white", hovermode="x unified", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_bench, use_container_width=True)
        except: st.error("Erreur indices")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Volatilité")
            if 'Delta' in df_g.columns:
                colors = ['#10b981' if v >= 0 else '#ef4444' for v in df_g['Delta']]
                fig_vol = go.Figure(go.Bar(x=df_g['Date'], y=df_g['Delta'], marker_color=colors))
                fig_vol.update_layout(template="simple_white", margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                st.plotly_chart(fig_vol, use_container_width=True)
        with c2:
            st.subheader("Plus-Value Cumulée")
            fig_pv = px.area(df_g, x='Date', y='Plus-value')
            fig_pv.update_traces(line_color='#10b981', fillcolor='rgba(16, 185, 129, 0.1)')
            fig_pv.update_layout(template="simple_white", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_pv, use_container_width=True)
    else:
        st.info("Historique vide.")

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
        st.plotly_chart(px.area(pd.DataFrame(res), x="Année", y="Capital", template="simple_white"), use_container_width=True)

with tab4:
    st.warning("Zone Admin. Modification directe BDD.")
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