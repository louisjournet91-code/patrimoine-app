import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# --- 1. CONFIGURATION & STYLE PREMIUM ---
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

# --- 2. GESTION BDD (STANDARD FRANÇAIS : POINT-VIRGULE) ---

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# Colonnes de votre fichier historique
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

def clean_french_number(x):
    """Nettoie les formats français (1 200,50 ou 0,5%)"""
    if isinstance(x, str):
        x = x.replace('%', '').replace(' ', '').replace(',', '.')
        if x.count('.') > 1: x = x.replace('.', '', x.count('.') - 1)
    try: return float(x)
    except: return 0.0

def load_state():
    # 1. Portefeuille
    if 'portfolio_df' not in st.session_state:
        if os.path.exists(FILE_PORTFOLIO):
            # On tente la lecture avec séparateur ; (Français) ou , (Anglais)
            try:
                df = pd.read_csv(FILE_PORTFOLIO, sep=';') 
                if len(df.columns) < 2: df = pd.read_csv(FILE_PORTFOLIO, sep=',') # Fallback
            except:
                df = pd.read_csv(FILE_PORTFOLIO, sep=',')
        else:
            df = pd.DataFrame(INITIAL_PORTFOLIO)
            df.to_csv(FILE_PORTFOLIO, index=False, sep=';') # On sauvegarde en ;
        
        df['Quantité'] = df['Quantité'].astype(float)
        df['PRU'] = df['PRU'].astype(float)
        st.session_state['portfolio_df'] = df

    # 2. Historique (Lecture forcée en ;)
    if os.path.exists(FILE_HISTORY):
        try:
            # C'est ici que se jouait l'erreur : on force sep=';'
            df_hist = pd.read_csv(FILE_HISTORY, sep=';', engine='python')
            
            # Si le fichier était encore en virgules, on retente
            if len(df_hist.columns) < 5:
                 df_hist = pd.read_csv(FILE_HISTORY, sep=',', engine='python')

            # Nettoyage
            for col in ['Total', 'PEA', 'BTC', 'Plus-value', 'PF_Return_TWR', 'ESE_Return']:
                if col in df_hist.columns:
                    df_hist[col] = df_hist[col].apply(lambda x: clean_french_number(x) if isinstance(x, str) else x)
            
            df_hist['Date'] = pd.to_datetime(df_hist['Date'], dayfirst=True, errors='coerce')
            df_hist = df_hist.dropna(subset=['Date'])
            
        except Exception as e:
            st.error(f"Erreur lecture historique (Vérifiez que le fichier utilise des points-virgules) : {e}")
            df_hist = pd.DataFrame(columns=HIST_COLS)
    else:
        df_hist = pd.DataFrame(columns=HIST_COLS)
    
    return df_hist

def save_portfolio():
    # Sauvegarde en séparateur POINT-VIRGULE pour Excel France
    st.session_state['portfolio_df'].to_csv(FILE_PORTFOLIO, index=False, sep=';')

def add_history_point(total, val_pea, val_btc, pv_totale):
    if os.path.exists(FILE_HISTORY):
        try:
            df = pd.read_csv(FILE_HISTORY, sep=';', engine='python')
        except:
            df = pd.read_csv(FILE_HISTORY, sep=',', engine='python')
    else:
        df = pd.DataFrame(columns=HIST_COLS)
        
    today = datetime.now().strftime("%d/%m/%Y")
    
    if today not in df['Date'].astype(str).values:
        new_data = {
            "Date": today,
            "Total": round(total, 2),
            "PEA": round(val_pea, 2),
            "BTC": round(val_btc, 2),
            "Plus-value": round(pv_totale, 2),
            # Valeurs par défaut pour combler les trous
            "Delta": 0, "PV_du_Jour": 0, "ESE": 0, "Flux_(€)": 0,
            "PF_Return_TWR": 0, "ESE_Return": 0, "PF_Index100": 0, "ESE_Index100": 0
        }
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
        # SAUVEGARDE AVEC SÉPARATEUR ;
        df.to_csv(FILE_HISTORY, index=False, sep=';') 
        return True
    return False

df_history_static = load_state()

# --- 3. MOTEUR TRANSACTIONNEL ---

def operation_tresorerie(amount):
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

def operation_trading(action, ticker, qty, price, nom="", type_a="Action"):
    df = st.session_state['portfolio_df']
    if ticker not in df['Ticker'].values:
        if action == "Vente": return False, "Inconnu"
        new_row = pd.DataFrame([{"Ticker": ticker, "Nom": nom, "Type": type_a, "Quantité": 0.0, "PRU": 0.0}])
        df = pd.concat([df, new_row], ignore_index=True)
    
    mask_c = df['Ticker'] == 'CASH'
    mask_a = df['Ticker'] == ticker
    cash = df.loc[mask_c, 'Quantité'].values[0]
    curr_q = df.loc[mask_a, 'Quantité'].values[0]
    curr_p = df.loc[mask_a, 'PRU'].values[0]
    total = qty * price
    
    if action == "Achat":
        if cash < total: return False, "Cash manquant"
        new_q = curr_q + qty
        new_p = ((curr_q * curr_p) + total) / new_q
        df.loc[mask_a, 'Quantité'] = new_q
        df.loc[mask_a, 'PRU'] = new_p
        df.loc[mask_c, 'Quantité'] = cash - total
    elif action == "Vente":
        if curr_q < qty: return False, "Pas assez de titres"
        df.loc[mask_a, 'Quantité'] = curr_q - qty
        df.loc[mask_c, 'Quantité'] = cash + total
        
    st.session_state['portfolio_df'] = df
    save_portfolio()
    return True, "Succès"

# --- 4. PRIX LIVE ---

@st.cache_data(ttl=60)
def get_prices(tickers):
    p = {"CASH": {"cur": 1.0, "prev": 1.0}}
    real = [t for t in tickers if t != "CASH"]
    if real:
        try:
            d = yf.download(real, period="5d", progress=False)['Close']
            if len(real) == 1:
                p[real[0]] = {"cur": float(d.iloc[-1]), "prev": float(d.iloc[-2])}
            else:
                l, pr = d.iloc[-1], d.iloc[-2]
                for t in real:
                    if t in l.index: p[t] = {"cur": float(l[t]), "prev": float(pr[t])}
        except: pass
    return p

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
cash_total = df[df['Ticker']=="CASH"]['Valo'].sum()
investi_titres = df[df['Ticker']!="CASH"]['Investi'].sum()
total_pv = total_pf - (investi_titres + cash_total)

# --- 5. INTERFACE ---

st.title("Gestion Patrimoniale Expert")
st.caption(f"Valo Live : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Patrimoine", f"{total_pf:,.2f} €")
k2.metric("Dont PEA", f"{val_pea:,.2f} €")
k3.metric("Dont Crypto", f"{val_btc:,.2f} €")
k4.metric("Plus-Value Totale", f"{total_pv:+,.2f} €")

st.markdown("---")

with st.sidebar:
    st.header("Guichet Unique")
    with st.expander("Opérations", expanded=True):
        typ = st.selectbox("Type", ["Apport Cash", "Achat", "Vente"])
        if typ == "Apport Cash":
            v = st.number_input("Montant", step=100.0)
            if st.button("Valider"): operation_tresorerie(v); st.rerun()
        else:
            tk = st.selectbox("Actif", [t for t in df['Ticker'].unique() if t!="CASH"])
            c1,c2 = st.columns(2)
            q = c1.number_input("Qte", 0.01)
            p = c2.number_input("Prix", 0.01)
            if st.button("Valider"): 
                ok, m = operation_trading(typ, tk, q, p)
                if ok: st.success(m); st.rerun()
    
    st.markdown("---")
    if st.button("💾 Sauvegarder État (Format Historique)"):
        res = add_history_point(total_pf, val_pea, val_btc, total_pv)
        if res: st.success("Ligne ajoutée au fichier historique !")
        else: st.warning("Ligne déjà présente pour aujourd'hui.")
        st.rerun()

tab1, tab2, tab3 = st.tabs(["📋 Positions", "📈 Historique", "🔮 Projection"])

with tab1:
    st.dataframe(df[['Nom', 'Quantité', 'PRU', 'Prix', 'Valo', 'Perf%']], 
                 column_config={"Perf%": st.column_config.ProgressColumn(min_value=-30, max_value=30, format="%.2f %%")},
                 use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Trajectoire Patrimoniale")
    if not df_history_static.empty:
        fig = px.area(df_history_static, x='Date', y='Total', title="Évolution Valeur Totale")
        fig.update_traces(line_color='#0f172a', fill_color='rgba(15, 23, 42, 0.1)')
        fig.update_layout(template="simple_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_history_static.sort_values('Date', ascending=False).head(5), use_container_width=True)
    else:
        st.info("Fichier historique vide ou format incorrect. Vérifiez les points-virgules.")

with tab3:
    st.subheader("Projection")
    col_in, col_out = st.columns([1, 2])
    with col_in:
        add = st.number_input("Apport/mois (€)", 500)
        rate = st.slider("Rendement (%)", 2.0, 12.0, 8.0)
        y = st.slider("Années", 5, 30, 15)
    with col_out:
        res = []
        cap = total_pf
        for i in range(1, y+1):
            cap = cap * (1 + rate/100) + (add*12)
            res.append({"Année": datetime.now().year+i, "Capital": cap})
        st.plotly_chart(px.area(pd.DataFrame(res), x="Année", y="Capital", template="simple_white"), use_container_width=True)