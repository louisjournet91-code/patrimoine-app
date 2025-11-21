import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Gestion Patrimoniale Expert", layout="wide", page_icon="🏛️")

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
    .cash-module { border-left: 5px solid #10b981; padding-left: 10px; }
    .trade-module { border-left: 5px solid #0f172a; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONSTANTES & FICHIERS ---

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

HIST_COLS = [
    "Date", "Total", "PEA", "BTC", "Plus-value", "Delta", "PV_du_Jour", 
    "ESE", "Flux_(€)", "PF_Return_TWR", "ESE_Return", 
    "PF_Index100", "ESE_Index100"
]

INITIAL_PORTFOLIO = {
    "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
    "Nom": ["BNP S&P 500", "Amundi World", "Lyxor Nasdaq", "Amundi USA x2", "Bitcoin", "Liquidités"],
    "Type": ["ETF Action", "ETF Action", "ETF Tech", "ETF Levier", "Crypto", "Cash"],
    "Quantité": [141.0, 716.0, 55.0, 176.0, 0.01, 510.84],
    "PRU": [24.41, 4.68, 71.73, 19.71, 90165.46, 1.00]
}

# --- 3. FONCTIONS ROBUSTES ---

def safe_float(x):
    """Nettoie n'importe quel format de nombre (1 200,50 ou 1200.50)"""
    if pd.isna(x) or x == "": return 0.0
    if isinstance(x, (float, int)): return float(x)
    if isinstance(x, str):
        # On garde uniquement chiffres, points, virgules, signe moins
        x = x.replace(' ', '').replace('%', '').replace('€', '')
        x = x.replace(',', '.')
        # Si plusieurs points (ex: 1.200.50), on garde le dernier
        if x.count('.') > 1:
            x = x.replace('.', '', x.count('.') - 1)
        try: return float(x)
        except: return 0.0
    return 0.0

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
            df_hist = pd.read_csv(FILE_HISTORY, sep=None, engine='python')
            # Nettoyage de TOUTES les colonnes numériques
            cols_num = ['Total', 'PEA', 'BTC', 'Plus-value', 'PV_du_Jour', 'PF_Index100', 'ESE_Index100']
            for col in cols_num:
                if col in df_hist.columns:
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
    """
    Sauvegarde avec Calcul du Delta pur : Valeur Actuelle - Valeur Veille (CSV)
    Sécurisée contre les virgules françaises.
    """
    # 1. Chargement de l'historique
    if os.path.exists(FILE_HISTORY):
        try: 
            # On lit sans forcer le type ici, le nettoyage se fera après
            df_hist = pd.read_csv(FILE_HISTORY, sep=None, engine='python')
        except: 
            df_hist = pd.DataFrame(columns=HIST_COLS)
    else:
        df_hist = pd.DataFrame(columns=HIST_COLS)

    # 2. Récupération de la valeur de LA VEILLE (J-1)
    # C'est ici que nous utilisons safe_float pour éviter le crash "ValueError"
    prev_total = 0.0
    prev_ese = 0.0
    prev_pf_idx = 100.0
    prev_ese_idx = 100.0
    prev_pv = 0.0
    
    if not df_hist.empty:
        last_row = df_hist.iloc[-1]
        # CORRECTION : Utilisation de safe_float au lieu de float
        prev_total = safe_float(last_row.get('Total', 0))
        prev_ese = safe_float(last_row.get('ESE', 0))
        prev_pf_idx = safe_float(last_row.get('PF_Index100', 100))
        prev_ese_idx = safe_float(last_row.get('ESE_Index100', 100))
        prev_pv = safe_float(last_row.get('Plus-value', 0))
    else:
        prev_total = total # Si fichier vide, on initialise avec la valeur actuelle

    # 3. Données Actuelles
    try:
        ese_price = df_pf.loc[df_pf['Ticker'].str.contains("ESE"), 'Prix_Actuel'].values[0]
    except:
        ese_price = 0.0
    
    flux = 0.0 

    # 4. LE CALCUL (Delta = Total - Veille)
    delta = total - prev_total 
    
    # La PV du jour est ce delta moins les flux éventuels
    pv_jour = delta - flux

    # TWR & Indices
    denom = prev_total + flux
    pf_return = (total - prev_total - flux) / denom if denom != 0 else 0.0
    
    # Calcul ESE Return
    if prev_ese != 0 and ese_price != 0:
        ese_return = (ese_price - prev_ese) / prev_ese
    else:
        ese_return = 0.0

    pf_index100 = prev_pf_idx * (1 + pf_return)
    ese_index100 = prev_ese_idx * (1 + ese_return)

    # 5. Sauvegarde
    today = datetime.now().strftime("%d/%m/%Y")
    
    # Conversion Date CSV pour vérifier doublon
    dates_existantes = []
    if not df_hist.empty and 'Date' in df_hist.columns:
        dates_existantes = df_hist['Date'].astype(str).values

    if today not in dates_existantes:
        new_data = {
            "Date": today,
            "Total": round(total, 2),
            "PEA": round(val_pea, 2),
            "BTC": round(val_btc, 2),
            "Plus-value": round(pv_totale, 2),
            "Delta": round(delta, 2),        
            "PV_du_Jour": round(pv_jour, 2),
            "ESE": round(ese_price, 2),
            "Flux_(€)": 0,
            "PF_Return_TWR": round(pf_return, 5),
            "ESE_Return": round(ese_return, 5),
            "PF_Index100": round(pf_index100, 2),
            "ESE_Index100": round(ese_index100, 2)
        }
        
        new_row = pd.DataFrame([new_data])
        df_final = pd.concat([df_hist, new_row], ignore_index=True)
        
        # Sauvegarde forcée avec point-virgule pour Excel France
        df_final.to_csv(FILE_HISTORY, index=False, sep=';')
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

df['Prix'] = df['Ticker'].apply(lambda x: market.get(x, {}).get("cur", df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Prev'] = df['Ticker'].apply(lambda x: market.get(x, {}).get("prev", df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Valo'] = df['Quantité'] * df['Prix']
df['Investi'] = df['Quantité'] * df['PRU']
df['PV'] = df['Valo'] - df['Investi']
df['Perf%'] = df.apply(lambda x: ((x['Prix']-x['PRU'])/x['PRU']*100) if x['PRU']>0 else 0, axis=1)
df['Var_Jour'] = df['Valo'] - (df['Quantité'] * df['Prev'])

val_btc = df[df['Ticker'].str.contains("BTC")]['Valo'].sum()
val_pea = df[~df['Ticker'].str.contains("BTC")]['Valo'].sum()
total_pf = df['Valo'].sum()
total_pv = df['PV'].sum()
volat_jour_live = df['Var_Jour'].sum() # La volatilité (P&L) du jour en live

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

# --- 6. INTERFACE ---
st.title("Gestion Patrimoniale Expert")
st.caption(f"Valo Live : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total", f"{total_pf:,.2f} €")
k2.metric("PEA", f"{val_pea:,.2f} €")
k3.metric("PV Latente", f"{total_pv:+,.2f} €")
k4.metric("Volatilité Jour", f"{volat_jour_live:+,.2f} €", help="Gain/Perte depuis la clôture d'hier")

st.markdown("---")

# --- 6. SIDEBAR (GUICHET) ---
with st.sidebar:
    st.header("Guichet Opérations")
    
    # Module 1 : Trésorerie
    with st.expander("💰 Trésorerie (Apport)", expanded=True):
        st.caption("Ajouter du Cash")
        mnt = st.number_input("Montant (€)", step=100.0)
        if st.button("Valider Virement", type="secondary", use_container_width=True):
            if mnt > 0:
                operation_tresorerie(mnt)
                st.success("Virement effectué !")
                st.rerun()

    st.markdown("---")

    # Module 2 : Trading
    with st.expander("📈 Trading (Ordres)", expanded=True):
        st.caption("Acheter / Vendre")
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
        qty = c1.number_input("Qté", 0.01)
        price = c2.number_input("Prix", 0.01)
        
        st.markdown(f"**Total : {qty*price:,.2f}€**")
        
        if st.button("Confirmer Ordre", type="primary", use_container_width=True):
            ok, msg = operation_trading(sens, tick, qty, price, nom)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    
    st.markdown("---")
    
    # BOUTON DE SAUVEGARDE (Mis à jour)
if st.button("💾 Sauvegarder Historique"):
        # On a retiré 'volat_jour_live' car le calcul se fait en interne maintenant
        succes, delta_calc = add_history_point(total_pf, val_pea, val_btc, total_pv, df)
        
        if succes: 
            st.success(f"Sauvegardé ! Delta vs Hier : {delta_calc:+.2f} €")
            import time; time.sleep(1); st.rerun()
        else: 
            st.warning("Déjà fait aujourd'hui")

tab1, tab2, tab3 = st.tabs(["Positions", "Analyse & Benchmarks", "Projection"])

with tab1:
    st.dataframe(df[['Nom','Quantité','PRU','Prix','Valo','Var_Jour','Perf%']], hide_index=True, use_container_width=True,
                 column_config={"Perf%": st.column_config.ProgressColumn(min_value=-30, max_value=30, format="%.2f %%"),
                                "Var_Jour": st.column_config.NumberColumn(format="%+.2f €")})

with tab2:
    if not df_history_static.empty:
        df_g = df_history_static.sort_values('Date').copy()
        
        # --- GRAPHIQUE 1 : BENCHMARK INDEX 100 ---
        st.subheader("Performance vs S&P 500 (Base 100)")
        fig_bench = go.Figure()
        # Ligne Portefeuille
        fig_bench.add_trace(go.Scatter(x=df_g['Date'], y=df_g['PF_Index100'], mode='lines', name='Mon Portefeuille', line=dict(color='#0f172a', width=3)))
        # Ligne Benchmark ESE
        fig_bench.add_trace(go.Scatter(x=df_g['Date'], y=df_g['ESE_Index100'], mode='lines', name='S&P 500 (ESE)', line=dict(color='#94a3b8', width=2, dash='dot')))
        fig_bench.update_layout(template="simple_white", hovermode="x unified")
        st.plotly_chart(fig_bench, use_container_width=True)

        c1, c2 = st.columns(2)
        
        # --- GRAPHIQUE 2 : HISTORIQUE PLUS-VALUE ---
        with c1:
            st.subheader("Évolution Plus-Value Latente")
            fig_pv = px.area(df_g, x='Date', y='Plus-value')
            fig_pv.update_traces(line_color='#10b981', fillcolor='rgba(16, 185, 129, 0.1)')
            fig_pv.update_layout(template="simple_white")
            st.plotly_chart(fig_pv, use_container_width=True)
            
        # --- GRAPHIQUE 3 : VOLATILITÉ (PV DU JOUR) ---
        with c2:
            st.subheader("Volatilité Quotidienne (P&L Jour)")
            # Couleurs dynamiques (Vert si > 0, Rouge si < 0)
            colors = ['#10b981' if v >= 0 else '#ef4444' for v in df_g['PV_du_Jour']]
            fig_vol = go.Figure(go.Bar(x=df_g['Date'], y=df_g['PV_du_Jour'], marker_color=colors))
            fig_vol.update_layout(template="simple_white", yaxis_title="Var. Journalière (€)")
            st.plotly_chart(fig_vol, use_container_width=True)
            
    else:
        st.info("Historique vide. Vérifiez que 'historique.csv' contient bien les colonnes PF_Index100 et ESE_Index100.")

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