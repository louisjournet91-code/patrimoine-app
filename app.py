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
            # Nettoyage
            for col in ['Total', 'PEA', 'BTC', 'Plus-value']:
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

def add_history_point(total, val_pea, val_btc, pv_totale):
    # Relecture propre avant ajout
    if os.path.exists(FILE_HISTORY):
        try: df = pd.read_csv(FILE_HISTORY, sep=None, engine='python')
        except: df = pd.DataFrame(columns=HIST_COLS)
    else:
        df = pd.DataFrame(columns=HIST_COLS)
        
    # Standardisation Date pour comparaison
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    today = pd.to_datetime(datetime.now().date())
    
    # Si la date d'aujourd'hui n'est PAS dans la colonne Date
    if not (df['Date'] == today).any():
        new_data = {
            "Date": today, # On stocke en objet datetime, pandas gèrera le format
            "Total": round(total, 2), "PEA": round(val_pea, 2), "BTC": round(val_btc, 2),
            "Plus-value": round(pv_totale, 2), "Delta": 0, "PV_du_Jour": 0, "ESE": 0, 
            "Flux_(€)": 0, "PF_Return_TWR": 0, "ESE_Return": 0, "PF_Index100": 0, "ESE_Index100": 0
        }
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Sauvegarde au format JJ/MM/AAAA pour Excel
        df.to_csv(FILE_HISTORY, index=False, sep=';', date_format='%d/%m/%Y')
        return True
    return False

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
df['Valo'] = df['Quantité'] * df['Prix']
df['Investi'] = df['Quantité'] * df['PRU']
df['PV'] = df['Valo'] - df['Investi']
df['Perf%'] = df.apply(lambda x: ((x['Prix']-x['PRU'])/x['PRU']*100) if x['PRU']>0 else 0, axis=1)

val_btc = df[df['Ticker'].str.contains("BTC")]['Valo'].sum()
val_pea = df[~df['Ticker'].str.contains("BTC")]['Valo'].sum()
total_pf = df['Valo'].sum()
total_pv = df['PV'].sum()

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
k3.metric("BTC", f"{val_btc:,.2f} €")
k4.metric("PV Latente", f"{total_pv:+,.2f} €")

st.markdown("---")

with st.sidebar:
    st.header("Guichet")
    with st.expander("Opérations", expanded=True):
        typ = st.selectbox("Type", ["Apport Cash", "Achat", "Vente"])
        if typ == "Apport Cash":
            v = st.number_input("Montant", step=100.0)
            if st.button("Valider"): op_cash(v); st.rerun()
        else:
            tk = st.selectbox("Actif", [t for t in df['Ticker'].unique() if t!="CASH"])
            c1,c2 = st.columns(2)
            q = c1.number_input("Qte", 0.01)
            p = c2.number_input("Prix", 0.01)
            if st.button("Valider"): 
                ok, m = op_trade(typ, tk, q, p)
                if ok: st.success(m); st.rerun()
                else: st.error(m)
    st.markdown("---")
    if st.button("💾 Sauvegarder Point Historique"):
        if add_history_point(total_pf, val_pea, val_btc, total_pv): st.success("Sauvegardé")
        else: st.warning("Déjà fait ajd")
        st.rerun()

tab1, tab2, tab3 = st.tabs(["Positions", "Historique", "Projection"])

with tab1:
    st.dataframe(df[['Nom','Quantité','PRU','Prix','Valo','Perf%']], hide_index=True, use_container_width=True)

with tab2:
    if not df_history_static.empty:
        # Préparation Graphique
        df_graph = df_history_static.copy()
        
        # Point Live
        live = pd.DataFrame([{"Date": pd.to_datetime(datetime.now().date()), "Total": total_pf}])
        
        # Concaténation et tri par date (CRITIQUE pour le tracé)
        df_final = pd.concat([df_graph[["Date", "Total"]], live], ignore_index=True)
        df_final = df_final.sort_values('Date') 
        
        # GRAPHIQUE (Correction fillcolor)
        fig = px.area(df_final, x='Date', y='Total', title="Trajectoire Totale")
        # LA CORRECTION EST ICI : fillcolor (pas fill_color)
        fig.update_traces(line_color='#0f172a', fillcolor='rgba(15, 23, 42, 0.1)')
        fig.update_layout(template="simple_white")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_history_static.sort_values('Date', ascending=False).head(5), use_container_width=True)
    else:
        st.info("Historique vide.")

with tab3:
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