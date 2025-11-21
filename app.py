import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# --- 1. CONFIGURATION & STYLE PREMIUM ---
st.set_page_config(page_title="Gestion Patrimoniale BDD", layout="wide", page_icon="🏛️")

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
    
    /* Style distinct pour les modules de saisie */
    .cash-module { border-left: 5px solid #10b981; padding-left: 10px; }
    .trade-module { border-left: 5px solid #0f172a; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. GESTION DE LA BASE DE DONNÉES (PERSISTANCE) ---

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# Colonnes exactes de votre fichier historique
HIST_COLS = [
    "Date", "Total", "PEA", "BTC", "Plus-value", "Delta", "PV_du_Jour", 
    "ESE", "Flux_(€)", "PF_Return_TWR", "ESE_Return", 
    "PF_Index100", "ESE_Index100"
]

# Vos données de départ
INITIAL_PORTFOLIO = {
    "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
    "Nom": ["BNP S&P 500", "Amundi World", "Lyxor Nasdaq", "Amundi USA x2", "Bitcoin", "Liquidités"],
    "Type": ["ETF Action", "ETF Action", "ETF Tech", "ETF Levier", "Crypto", "Cash"],
    "Quantité": [141.0, 716.0, 55.0, 176.0, 0.01, 510.84],
    "PRU": [24.41, 4.68, 71.73, 19.71, 90165.46, 1.00]
}

def clean_french_number(x):
    """Convertit '0,05%' ou '1 200,50' en float Python"""
    if isinstance(x, str):
        x = x.replace('%', '').replace(' ', '').replace(',', '.')
        # Gestion du cas où il y aurait plusieurs points (ex: 1.200.50)
        if x.count('.') > 1:
            x = x.replace('.', '', x.count('.') - 1)
    try:
        return float(x)
    except:
        return 0.0

def load_state():
    """Charge les données depuis CSV vers Session State."""
    
    # 1. Portefeuille
    if 'portfolio_df' not in st.session_state:
        if os.path.exists(FILE_PORTFOLIO):
            df = pd.read_csv(FILE_PORTFOLIO)
        else:
            df = pd.DataFrame(INITIAL_PORTFOLIO)
            df.to_csv(FILE_PORTFOLIO, index=False)
        
        df['Quantité'] = df['Quantité'].astype(float)
        df['PRU'] = df['PRU'].astype(float)
        st.session_state['portfolio_df'] = df

    # 2. Historique (Lecture intelligente format Français)
    if os.path.exists(FILE_HISTORY):
        try:
            # On essaie de lire avec détection auto du séparateur
            df_hist = pd.read_csv(FILE_HISTORY, sep=None, engine='python')
            
            # Nettoyage des colonnes numériques
            cols_to_clean = ['Total', 'PEA', 'BTC', 'Plus-value']
            for col in cols_to_clean:
                if col in df_hist.columns:
                    df_hist[col] = df_hist[col].apply(lambda x: clean_french_number(x) if isinstance(x, str) else x)
            
            # Conversion Date
            df_hist['Date'] = pd.to_datetime(df_hist['Date'], dayfirst=True, errors='coerce')
            df_hist = df_hist.dropna(subset=['Date']) 
            
        except Exception as e:
            st.error(f"Erreur lecture historique : {e}")
            df_hist = pd.DataFrame(columns=HIST_COLS)
    else:
        df_hist = pd.DataFrame(columns=HIST_COLS)
    
    return df_hist

def save_portfolio():
    """Sauvegarde la Session State vers le CSV"""
    st.session_state['portfolio_df'].to_csv(FILE_PORTFOLIO, index=False)

def add_history_point(total, val_pea, val_btc, pv_totale):
    """Ajoute une ligne respectant VOTRE format strict"""
    if os.path.exists(FILE_HISTORY):
        df = pd.read_csv(FILE_HISTORY, sep=None, engine='python')
    else:
        df = pd.DataFrame(columns=HIST_COLS)
        
    today = datetime.now().strftime("%d/%m/%Y")
    
    # Vérification doublon date
    if today not in df['Date'].astype(str).values:
        new_data = {
            "Date": today,
            "Total": round(total, 2),
            "PEA": round(val_pea, 2),
            "BTC": round(val_btc, 2),
            "Plus-value": round(pv_totale, 2),
            # Champs calculés mis à 0 par défaut pour compatibilité
            "Delta": 0, "PV_du_Jour": 0, "ESE": 0, "Flux_(€)": 0,
            "PF_Return_TWR": 0, "ESE_Return": 0, 
            "PF_Index100": 0, "ESE_Index100": 0
        }
        
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(FILE_HISTORY, index=False)
        return True
    return False

# Chargement au démarrage
df_history_static = load_state()

# --- 3. MOTEUR TRANSACTIONNEL ROBUSTE ---

def operation_tresorerie(amount):
    """Gère le Cash"""
    df = st.session_state['portfolio_df']
    mask = df['Ticker'] == 'CASH'
    
    if not mask.any():
         new_cash = pd.DataFrame([{"Ticker": "CASH", "Nom": "Liquidités", "Type": "Cash", "Quantité": 0.0, "PRU": 1.0}])
         df = pd.concat([df, new_cash], ignore_index=True)
         mask = df['Ticker'] == 'CASH'

    current_cash = df.loc[mask, 'Quantité'].values[0]
    df.loc[mask, 'Quantité'] = current_cash + amount
    
    st.session_state['portfolio_df'] = df
    save_portfolio() # Persistance
    return True

def operation_trading(action, ticker, qty, price, nom_actif="Nouvel Actif", type_actif="Action"):
    """Gère le Trading"""
    df = st.session_state['portfolio_df']
    
    # Gestion Création Actif
    if ticker not in df['Ticker'].values:
        if action == "Vente": return False, "Actif inconnu"
        new_row = pd.DataFrame([{
            "Ticker": ticker, "Nom": nom_actif, "Type": type_actif, 
            "Quantité": 0.0, "PRU": 0.0
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    
    mask_cash = df['Ticker'] == 'CASH'
    cash_dispo = df.loc[mask_cash, 'Quantité'].values[0]
    total_ordre = qty * price
    
    mask_asset = df['Ticker'] == ticker
    current_qty = df.loc[mask_asset, 'Quantité'].values[0]
    current_pru = df.loc[mask_asset, 'PRU'].values[0]
    
    if action == "Achat":
        if cash_dispo < total_ordre:
            return False, f"Manque de liquidités ({total_ordre - cash_dispo:.2f}€)"
        
        new_qty = current_qty + qty
        new_pru = ((current_qty * current_pru) + total_ordre) / new_qty
        
        df.loc[mask_asset, 'Quantité'] = new_qty
        df.loc[mask_asset, 'PRU'] = new_pru
        df.loc[mask_cash, 'Quantité'] = cash_dispo - total_ordre
        
    elif action == "Vente":
        if current_qty < qty: return False, "Quantité insuffisante"
        
        df.loc[mask_asset, 'Quantité'] = current_qty - qty
        df.loc[mask_cash, 'Quantité'] = cash_dispo + total_ordre
        
    st.session_state['portfolio_df'] = df
    save_portfolio() # Persistance
    return True, "Ordre exécuté avec succès"

# --- 4. PRIX & CALCULS TEMPS RÉEL ---

@st.cache_data(ttl=60)
def get_prices(tickers):
    prices = {"CASH": {"cur": 1.0, "prev": 1.0}}
    real = [t for t in tickers if t != "CASH"]
    if real:
        try:
            data = yf.download(real, period="5d", progress=False)['Close']
            if len(real) == 1:
                prices[real[0]] = {"cur": float(data.iloc[-1]), "prev": float(data.iloc[-2])}
            else:
                last = data.iloc[-1]
                prev = data.iloc[-2]
                for t in real:
                    if t in last.index:
                        prices[t] = {"cur": float(last[t]), "prev": float(prev[t])}
        except: pass
    return prices

# Récupération du DF actif
df = st.session_state['portfolio_df'].copy()
market_data = get_prices(df['Ticker'].unique())

# Application Prix
df['Prix_Actuel'] = df['Ticker'].apply(lambda x: market_data.get(x, {}).get("cur", df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Prix_Veille'] = df['Ticker'].apply(lambda x: market_data.get(x, {}).get("prev", df.loc[df['Ticker']==x, 'PRU'].values[0]))

# Calculs Financiers
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['Plus_Value'] = df['Valo'] - df['Investi']
df['Perf_%'] = df.apply(lambda x: ((x['Prix_Actuel'] - x['PRU']) / x['PRU'] * 100) if x['PRU']>0 else 0, axis=1)
df['Var_Jour_€'] = df['Valo'] - (df['Quantité'] * df['Prix_Veille'])

# Totaux ventilés (Pour historique)
val_btc = df[df['Ticker'].str.contains("BTC")]['Valo'].sum()
val_pea = df[~df['Ticker'].str.contains("BTC")]['Valo'].sum()
total_pf = df['Valo'].sum()
cash_total = df[df['Ticker']=="CASH"]['Valo'].sum()
investi_titres = df[df['Ticker']!="CASH"]['Investi'].sum()
total_pv = total_pf - (investi_titres + cash_total)

# --- 5. INTERFACE PRINCIPALE ---

st.title("Gestion Patrimoniale & BDD")
st.caption(f"Données Persistantes (CSV) • {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Patrimoine Total", f"{total_pf:,.2f} €")
k2.metric("Dont PEA (Cash inclus)", f"{val_pea:,.2f} €")
k3.metric("Dont BTC", f"{val_btc:,.2f} €")
k4.metric("PV Latente", f"{total_pv:+,.2f} €", f"{(total_pv/investi_titres)*100 if investi_titres>0 else 0:+.2f}%")

st.markdown("---")

# --- 6. SIDEBAR (GUICHET) ---
with st.sidebar:
    st.header("Guichet Opérations")
    
    # Module 1 : Trésorerie
    with st.expander("💰 Trésorerie (Apport)", expanded=True):
        mnt = st.number_input("Montant (€)", step=100.0)
        if st.button("Valider Virement", type="secondary", use_container_width=True):
            if mnt > 0:
                operation_tresorerie(mnt)
                st.success("Virement effectué !")
                st.rerun()

    st.markdown("---")

    # Module 2 : Trading
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
        qty = c1.number_input("Qté", 0.01)
        price = c2.number_input("Prix", 0.01)
        
        st.write(f"**Total : {qty*price:,.2f}€**")
        
        if st.button("Confirmer Ordre", type="primary", use_container_width=True):
            ok, msg = operation_trading(sens, tick, qty, price, nom)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    
    st.markdown("---")
    # Bouton Historique
    if st.button("💾 Sauvegarder Point Historique"):
        res = add_history_point(total_pf, val_pea, val_btc, total_pv)
        if res: st.success("Ligne ajoutée au fichier historique !")
        else: st.warning("Point déjà existant pour aujourd'hui.")
        st.rerun()

# --- 7. ONGLETS D'ANALYSE ---

tab1, tab2, tab3, tab4 = st.tabs(["📋 Portefeuille", "📊 Analyse", "⏳ Historique", "🔮 Projection"])

with tab1:
    st.dataframe(
        df[['Nom', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo', 'Var_Jour_€', 'Perf_%']],
        column_config={
            "Perf_%": st.column_config.ProgressColumn("Perf.", format="%+.2f %%", min_value=-30, max_value=30),
            "Valo": st.column_config.NumberColumn(format="%.2f €"),
            "Var_Jour_€": st.column_config.NumberColumn(format="%+.2f €"),
        },
        hide_index=True, use_container_width=True
    )

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig_w = go.Figure(go.Waterfall(
            measure=["relative"]*len(df), x=df['Nom'], y=df['Plus_Value'],
            connector={"line":{"color":"#cbd5e1"}},
            decreasing={"marker":{"color":"#ef4444"}}, increasing={"marker":{"color":"#10b981"}}
        ))
        fig_w.update_layout(title="Contribution P&L (€)", template="simple_white")
        st.plotly_chart(fig_w, use_container_width=True)
    with c2:
        fig_d = px.pie(df[df['Ticker']!="CASH"], values='Valo', names='Nom', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_d.update_layout(title="Allocation Actifs", showlegend=False, template="simple_white")
        st.plotly_chart(fig_d, use_container_width=True)

with tab3:
    st.subheader("Trajectoire Patrimoniale")
    # Fusion Historique CSV + Point Live
    if not df_history_static.empty:
        current_point = pd.DataFrame([{
            "Date": datetime.now(), 
            "Total": total_pf, 
            "PEA": val_pea,
            "BTC": val_btc
        }])
        
        # S'assurer que les colonnes matchent pour la concaténation
        cols_graph = ["Date", "Total"]
        df_graph = pd.concat([df_history_static[cols_graph], current_point[cols_graph]], ignore_index=True)
        
        fig_hist = px.area(df_graph, x='Date', y='Total', title="Passé + Présent", color_discrete_sequence=["#0f172a"])
        fig_hist.update_layout(template="simple_white")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Affichage tableau brut
        st.caption("Données brutes (Derniers points)")
        st.dataframe(df_history_static.sort_values('Date', ascending=False).head(5), use_container_width=True)
    else:
        st.info("Historique vide. Cliquez sur le bouton 'Sauvegarder Point Historique' dans la barre latérale.")

with tab4:
    st.subheader("Futur")
    col_in, col_out = st.columns([1, 2])
    with col_in:
        add = st.number_input("Apport mensuel (€)", 500)
        rate = st.slider("Rendement (%)", 2.0, 15.0, 8.0)
        years = st.slider("Horizon (ans)", 5, 30, 15)
    with col_out:
        proj = []
        cap = total_pf
        for y in range(1, years+1):
            cap = cap * (1 + rate/100) + (add*12)
            proj.append({"Année": datetime.now().year+y, "Capital": cap})
        st.plotly_chart(px.area(pd.DataFrame(proj), x="Année", y="Capital", template="simple_white"), use_container_width=True)
        st.success(f"Capital à terme : {cap:,.0f} €")