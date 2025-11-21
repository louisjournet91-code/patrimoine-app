import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import io

# --- 1. CONFIGURATION & STYLE PREMIUM ---
st.set_page_config(page_title="Ultimate Liquid Estate", layout="wide", page_icon="💎")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');

    /* FOND HAUTE DÉFINITION */
    .stApp {
        background-color: #f0f4f8;
        background-image: 
            radial-gradient(at 0% 0%, hsla(210,100%,96%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(220,100%,93%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(190,100%,94%,1) 0, transparent 50%),
            radial-gradient(at 0% 100%, hsla(260,100%,95%,1) 0, transparent 50%);
        font-family: 'Outfit', sans-serif;
        color: #1e293b;
    }

    /* GLASSMORPHISM RAFFINÉ */
    div[data-testid="stMetric"], div[data-testid="stDataFrame"], div.stPlotlyChart, div.stForm, div.stExpander {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(25px) saturate(180%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
        padding: 20px !important;
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-3px); }

    /* TYPOGRAPHIE */
    h1, h2, h3 { color: #0f172a; font-weight: 800; letter-spacing: -0.5px; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 32px; font-weight: 800; background: linear-gradient(135deg, #0f172a 0%, #334155 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    /* SIDEBAR & BOUTONS */
    section[data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.5); backdrop-filter: blur(40px); border-right: 1px solid rgba(255,255,255,0.6); }
    div.stButton > button { border-radius: 14px; background: #0f172a; color: white; font-weight: 600; border: none; padding: 0.5rem 1rem; transition: all 0.2s; }
    div.stButton > button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(15, 23, 42, 0.2); }
</style>
""", unsafe_allow_html=True)

# --- 2. GESTION DES DONNÉES ROBUSTE ---

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# Structure initiale si fichiers absents
INITIAL_PORTFOLIO = {
    "Ticker": ["ESE.PA", "CW8.PA", "BTC-EUR", "CASH"],
    "Nom": ["S&P 500 BNP", "MSCI World", "Bitcoin", "Liquidités"],
    "Type": ["ETF Action", "ETF Action", "Crypto", "Cash"],
    "Quantité": [0.0, 0.0, 0.0, 1000.0],
    "PRU": [0.0, 0.0, 0.0, 1.0]
}

def safe_float(x):
    """Conversion sécurisée en float pour éviter les crashs."""
    if pd.isna(x) or str(x).strip() == "": return 0.0
    s = str(x).strip().replace('"', '').replace('%', '').replace('€', '').replace(' ', '').replace(',', '.')
    try: return float(s)
    except: return 0.0

def load_state():
    """Chargement et réparation automatique des fichiers."""
    # 1. Portefeuille
    if 'portfolio_df' not in st.session_state:
        if os.path.exists(FILE_PORTFOLIO):
            try:
                df = pd.read_csv(FILE_PORTFOLIO, sep=';')
                if df.shape[1] < 2: df = pd.read_csv(FILE_PORTFOLIO, sep=',') # Fallback separateur
            except: df = pd.DataFrame(INITIAL_PORTFOLIO)
        else:
            df = pd.DataFrame(INITIAL_PORTFOLIO)
            df.to_csv(FILE_PORTFOLIO, index=False, sep=';')
        
        # Nettoyage des types
        for col in ['Quantité', 'PRU']:
            df[col] = df[col].apply(safe_float)
        st.session_state['portfolio_df'] = df

    # 2. Historique
    if os.path.exists(FILE_HISTORY):
        try:
            with open(FILE_HISTORY, 'r') as f: raw_data = f.read()
            df_hist = pd.read_csv(io.StringIO(raw_data), sep=',', engine='python')
            df_hist['Date'] = pd.to_datetime(df_hist['Date'], dayfirst=True, errors='coerce')
            df_hist = df_hist.dropna(subset=['Date'])
            # Conversion colonnes numériques
            cols_num = [c for c in df_hist.columns if c != "Date"]
            for c in cols_num: df_hist[c] = df_hist[c].apply(safe_float)
            return df_hist
        except: pass
    return pd.DataFrame(columns=["Date", "Total", "Delta"])

def save_portfolio():
    st.session_state['portfolio_df'].to_csv(FILE_PORTFOLIO, index=False, sep=';')

# --- 3. LOGIQUE FINANCIÈRE ---

@st.cache_data(ttl=300) # Cache de 5 minutes pour ne pas spammer Yahoo
def get_prices(tickers):
    prices = {"CASH": {"cur": 1.0, "prev": 1.0}}
    real_tickers = [t for t in tickers if t != "CASH" and isinstance(t, str)]
    
    if not real_tickers: return prices

    try:
        # Téléchargement en bloc plus efficace
        data = yf.download(real_tickers, period="5d", progress=False)['Close']
        
        # Gestion si un seul ticker (retourne Series) ou plusieurs (DataFrame)
        if len(real_tickers) == 1:
            data = data.to_frame(name=real_tickers[0])
            
        if not data.empty:
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2] if len(data) > 1 else last_row
            
            for t in real_tickers:
                if t in last_row.index:
                    # Gestion des NaN
                    cur = float(last_row[t]) if not pd.isna(last_row[t]) else 0.0
                    prev = float(prev_row[t]) if not pd.isna(prev_row[t]) else cur
                    prices[t] = {"cur": cur, "prev": prev}
    except Exception as e:
        st.toast(f"Erreur API Yahoo: {e}", icon="⚠️")
        
    return prices

# --- 4. FONCTIONS OPÉRATIONNELLES ---

def operation_tresorerie(amount):
    """Ajoute ou retire du cash du portefeuille."""
    df = st.session_state['portfolio_df']
    
    # Vérifier si la ligne CASH existe, sinon la créer
    if 'CASH' not in df['Ticker'].values:
        new_row = pd.DataFrame([{"Ticker": "CASH", "Nom": "Liquidités", "Type": "Cash", "Quantité": 0.0, "PRU": 1.0}])
        df = pd.concat([df, new_row], ignore_index=True)
    
    mask = df['Ticker'] == 'CASH'
    current_qty = df.loc[mask, 'Quantité'].values[0]
    
    # Mise à jour
    df.loc[mask, 'Quantité'] = current_qty + amount
    df.loc[mask, 'PRU'] = 1.0 # Le cash vaut toujours 1
    
    st.session_state['portfolio_df'] = df
    save_portfolio()

def op_trade(sens, tick, q, p, nom=""):
    df = st.session_state['portfolio_df']
    
    # Init ticker si inconnu
    if tick not in df['Ticker'].values:
        if sens == "Vente": return False, "Actif inconnu, impossible de vendre."
        new_row = pd.DataFrame([{"Ticker": tick, "Nom": nom, "Type": "Action", "Quantité": 0.0, "PRU": 0.0}])
        df = pd.concat([df, new_row], ignore_index=True)
    
    mask_asset = df['Ticker'] == tick
    mask_cash = df['Ticker'] == 'CASH'
    
    # Sécurité Cash
    if 'CASH' not in df['Ticker'].values: operation_tresorerie(0)
    
    cash_dispo = df.loc[mask_cash, 'Quantité'].values[0]
    cur_qty = df.loc[mask_asset, 'Quantité'].values[0]
    cur_pru = df.loc[mask_asset, 'PRU'].values[0]
    total_op = q * p

    if sens == "Achat":
        if cash_dispo < total_op: return False, "Trésorerie insuffisante."
        # Calcul du PAMP (Prix Moyen Pondéré)
        new_qty = cur_qty + q
        new_pru = ((cur_qty * cur_pru) + total_op) / new_qty
        
        df.loc[mask_asset, 'Quantité'] = new_qty
        df.loc[mask_asset, 'PRU'] = new_pru
        df.loc[mask_cash, 'Quantité'] = cash_dispo - total_op
        
    elif sens == "Vente":
        if cur_qty < q: return False, "Quantité de titres insuffisante."
        df.loc[mask_asset, 'Quantité'] = cur_qty - q
        # PRU ne change pas à la vente fiscale en France (généralement)
        df.loc[mask_cash, 'Quantité'] = cash_dispo + total_op

    st.session_state['portfolio_df'] = df
    save_portfolio()
    return True, "Ordre exécuté avec succès."

def add_history_point(metrics):
    """Enregistre l'état actuel dans l'historique."""
    df_hist = load_state()
    today = datetime.now().strftime("%d/%m/%Y")
    
    # Eviter doublons jour même
    dates_existantes = []
    if not df_hist.empty and 'Date' in df_hist.columns:
        dates_existantes = df_hist['Date'].dt.strftime("%d/%m/%Y").values
        
    if today in dates_existantes:
        return False, 0.0

    prev_total = df_hist.iloc[-1]['Total'] if not df_hist.empty else metrics['Total']
    delta = metrics['Total'] - prev_total
    
    new_row = {
        "Date": today,
        "Total": round(metrics['Total'], 2),
        "Delta": round(delta, 2),
        "PEA": round(metrics['PEA'], 2),
        "Crypto": round(metrics['Crypto'], 2),
        "Cash": round(metrics['Cash'], 2)
    }
    
    # Sauvegarde CSV manuelle pour éviter les bugs de format
    line = ",".join([str(v) for v in new_row.values()])
    mode = 'a' if os.path.exists(FILE_HISTORY) else 'w'
    with open(FILE_HISTORY, mode) as f:
        if mode == 'w': f.write(",".join(new_row.keys()) + "\n")
        else: f.write("\n")
        f.write(line)
        
    return True, delta

# --- 5. CALCULS & PRÉPARATION ---

df_hist_static = load_state()
df = st.session_state['portfolio_df'].copy()

# Récupération Prix Marché
market_prices = get_prices(df['Ticker'].unique())

# Enrichissement DataFrame
df['Prix_Actuel'] = df['Ticker'].apply(lambda x: market_prices.get(x, {}).get("cur", df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Prev_Price'] = df['Ticker'].apply(lambda x: market_prices.get(x, {}).get("prev", df.loc[df['Ticker']==x, 'PRU'].values[0]))

df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['PV_Latente'] = df['Valo'] - df['Investi']
df['Perf_%'] = df.apply(lambda x: ((x['Prix_Actuel']-x['PRU'])/x['PRU']*100) if x['PRU']>0 else 0, axis=1)
df['Var_Jour_€'] = df['Valo'] - (df['Quantité'] * df['Prev_Price'])

# KPIs Globaux
total_pf = df['Valo'].sum()
cash_val = df[df['Ticker'] == 'CASH']['Valo'].sum()
crypto_val = df[df['Type'] == 'Crypto']['Valo'].sum()
pea_val = total_pf - cash_val - crypto_val
total_pv = df['PV_Latente'].sum()

metrics_dict = {"Total": total_pf, "PEA": pea_val, "Crypto": crypto_val, "Cash": cash_val}

# Variation journalière (Live vs Hier Clôture)
var_jour_live = df['Var_Jour_€'].sum()

# --- 6. INTERFACE BENTO ---

col_head1, col_head2 = st.columns([1, 4])
with col_head2:
    st.title("🏛️ Patrimoine")
    st.caption(f"Mise à jour : {datetime.now().strftime('%d %B %Y • %H:%M')}")

# HERO METRIC
st.markdown(f"""
<div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
            padding: 30px; border-radius: 24px; border: 1px solid white; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.03); margin-bottom: 20px; text-align: center;">
    <p style="color: #64748b; font-size: 13px; text-transform: uppercase; letter-spacing: 2px;">Valeur Nette Totale</p>
    <h1 style="font-size: 60px; margin: 0; background: -webkit-linear-gradient(45deg, #1e293b, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        {total_pf:,.2f} €
    </h1>
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">
        <span style="color: {'#10b981' if var_jour_live >= 0 else '#ef4444'}; font-weight: 600;">
            {var_jour_live:+.2f} € (24h)
        </span>
        <span style="color: {'#10b981' if total_pv >= 0 else '#ef4444'}; font-weight: 600;">
            Total PV: {total_pv:+.2f} €
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# DASHBOARD GRID
c1, c2, c3 = st.columns(3)
c1.metric("Liquidités", f"{cash_val:,.2f} €", f"{(cash_val/total_pf)*100:.1f} % Alloc.")
c2.metric("Actions (PEA/CTO)", f"{pea_val:,.2f} €", f"{(pea_val/total_pf)*100:.1f} % Alloc.")
c3.metric("Crypto-actifs", f"{crypto_val:,.2f} €", f"{(crypto_val/total_pf)*100:.1f} % Alloc.")

st.markdown("---")

# SIDEBAR DE CONTRÔLE
with st.sidebar:
    st.header("🕹️ Opérations")
    
    with st.expander("💶 Mouvements Cash", expanded=True):
        montant_flux = st.number_input("Montant (€)", step=100.0, help="Positif pour dépôt, Négatif pour retrait")
        if st.button("Exécuter Flux", type="secondary"):
            if montant_flux != 0:
                operation_tresorerie(montant_flux)
                st.success("Flux enregistré !")
                st.rerun()
            else:
                st.warning("Montant nul.")

    with st.expander("📈 Salle de Marché", expanded=True):
        sens = st.radio("Sens", ["Achat", "Vente"], horizontal=True)
        type_asset = st.radio("Actif", ["Existant", "Nouveau"], horizontal=True, label_visibility="collapsed")
        
        if type_asset == "Existant":
            # On exclut le CASH de la liste de trading
            tick = st.selectbox("Ticker", [t for t in df['Ticker'].unique() if t != 'CASH'])
            nom = df.loc[df['Ticker']==tick, 'Nom'].values[0] if tick else ""
        else:
            tick = st.text_input("Symbole (ex: TTE.PA)").upper()
            nom = st.text_input("Nom de l'actif")
            
        col_q, col_p = st.columns(2)
        qty = col_q.number_input("Quantité", min_value=0.000001, step=1.0, format="%.6f")
        price = col_p.number_input("Prix Unitaire", min_value=0.01, step=0.1)
        
        if st.button("Valider Ordre", type="primary"):
            ok, msg = op_trade(sens, tick, qty, price, nom)
            if ok: 
                st.success(msg)
                st.rerun()
            else: 
                st.error(msg)

    st.markdown("---")
    if st.button("💾 Sauvegarder Point Historique"):
        succes, d = add_history_point(metrics_dict)
        if succes: 
            st.success(f"Sauvegardé. Variation: {d:+.2f}€")
            st.rerun()
        else: 
            st.warning("Point déjà existant pour aujourd'hui.")

# VUES DÉTAILLÉES
tab1, tab2, tab3 = st.tabs(["📊 Analyse Visuelle", "📋 Détail Positions", "🔧 Données Brutes"])

with tab1:
    g1, g2 = st.columns([2, 1])
    
    with g1:
        st.subheader("Trajectoire Patrimoniale")
        if not df_hist_static.empty and 'Total' in df_hist_static.columns:
            # Graphique Area Chart Clean
            fig = px.area(df_hist_static, x='Date', y='Total', color_discrete_sequence=['#3b82f6'])
            fig.update_layout(
                template="simple_white", 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0,r=0,t=10,b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("En attente de données historiques (Sauvegardez via la Sidebar)")

    with g2:
        st.subheader("Allocation Actuelle")
        # Graphique Donut
        alloc_df = df.groupby("Type")['Valo'].sum().reset_index()
        fig_donut = px.pie(alloc_df, values='Valo', names='Type', hole=0.6, 
                           color_discrete_sequence=px.colors.sequential.RdBu)
        fig_donut.update_layout(
            showlegend=False, 
            margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text='Répartition', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

with tab2:
    # Tableau stylisé
    st.dataframe(
        df[['Nom', 'Ticker', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo', 'Perf_%', 'Var_Jour_€']],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Nom": st.column_config.TextColumn("Actif", width="medium"),
            "Quantité": st.column_config.NumberColumn("Qté", format="%.4f"),
            "PRU": st.column_config.NumberColumn("PRU", format="%.2f €"),
            "Prix_Actuel": st.column_config.NumberColumn("Prix", format="%.2f €"),
            "Valo": st.column_config.NumberColumn("Valorisation", format="%.2f €"),
            "Perf_%": st.column_config.ProgressColumn("Perf %", format="%.2f %%", min_value=-20, max_value=20),
            "Var_Jour_€": st.column_config.NumberColumn("Var. Jour", format="%+.2f €")
        }
    )

with tab3:
    st.write("Édition manuelle (Attention)")
    edited_df = st.data_editor(df, num_rows="dynamic")
    if st.button("Forcer Sauvegarde CSV"):
        edited_df.to_csv(FILE_PORTFOLIO, index=False, sep=';')
        st.success("Fichier écrasé avec les données ci-dessus.")