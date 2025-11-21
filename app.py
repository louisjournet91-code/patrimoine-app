import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="Gestion Patrimoniale Active", layout="wide", page_icon="🏛️")

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
    /* Style pour le formulaire d'achat/vente */
    div[data-testid="stForm"] { border: 1px solid #d4af37; background-color: #fffdf5; }
</style>
""", unsafe_allow_html=True)

# --- 2. INITIALISATION DES DONNÉES (MÉMOIRE) ---

# Données de base (Votre situation officielle)
INITIAL_PORTFOLIO = {
    "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
    "Nom": ["BNP S&P 500", "Amundi World", "Lyxor Nasdaq", "Amundi USA x2", "Bitcoin", "Liquidités"],
    "Type": ["ETF Action", "ETF Action", "ETF Tech", "ETF Levier", "Crypto", "Cash"],
    "Quantité": [141.0, 716.0, 55.0, 176.0, 0.01, 510.84],
    "PRU": [24.41, 4.68, 71.73, 19.71, 90165.46, 1.00]
}

# Initialisation de la Session State (La mémoire vive du site)
if 'portfolio_df' not in st.session_state:
    st.session_state['portfolio_df'] = pd.DataFrame(INITIAL_PORTFOLIO)
    # On force le type float pour éviter les erreurs de calcul
    st.session_state['portfolio_df']['Quantité'] = st.session_state['portfolio_df']['Quantité'].astype(float)
    st.session_state['portfolio_df']['PRU'] = st.session_state['portfolio_df']['PRU'].astype(float)

# --- 3. FONCTIONS DE GESTION (ACHAT/VENTE) ---

def update_cash(amount):
    """Met à jour la ligne CASH"""
    idx = st.session_state['portfolio_df'].index[st.session_state['portfolio_df']['Ticker'] == "CASH"].tolist()[0]
    current_cash = st.session_state['portfolio_df'].at[idx, 'Quantité']
    new_cash = current_cash + amount
    st.session_state['portfolio_df'].at[idx, 'Quantité'] = new_cash
    return new_cash

def execute_order(action, ticker, qty, price):
    df = st.session_state['portfolio_df']
    
    # Vérifier si l'actif existe, sinon l'ajouter (Sauf si c'est Cash)
    if ticker not in df['Ticker'].values:
        new_row = {"Ticker": ticker, "Nom": ticker, "Type": "Autre", "Quantité": 0.0, "PRU": 0.0}
        st.session_state['portfolio_df'] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = st.session_state['portfolio_df'] # Recharger

    idx = df.index[df['Ticker'] == ticker].tolist()[0]
    current_qty = df.at[idx, 'Quantité']
    current_pru = df.at[idx, 'PRU']
    
    transaction_total = qty * price
    
    if action == "Achat":
        # 1. Vérification Liquidité
        cash_idx = df.index[df['Ticker'] == "CASH"].tolist()[0]
        cash_dispo = df.at[cash_idx, 'Quantité']
        
        if cash_dispo < transaction_total:
            return False, "❌ Liquidités insuffisantes pour cet achat !"
            
        # 2. Calcul Nouveau PRU (Moyenne Pondérée)
        # Formule : (AncienneValo + NouvelAchat) / NouvelleQte
        new_qty = current_qty + qty
        new_pru = ((current_qty * current_pru) + (qty * price)) / new_qty
        
        # 3. Mise à jour Actif
        df.at[idx, 'Quantité'] = new_qty
        df.at[idx, 'PRU'] = new_pru
        
        # 4. Débit Cash
        update_cash(-transaction_total)
        return True, f"✅ Achat validé : {qty} {ticker} à {price}€ (Nouveau PRU: {new_pru:.2f}€)"

    elif action == "Vente":
        if current_qty < qty:
            return False, "❌ Vous ne possédez pas assez de titres !"
            
        # 1. Mise à jour Quantité (Le PRU ne change pas à la vente fiscalement)
        df.at[idx, 'Quantité'] = current_qty - qty
        
        # 2. Crédit Cash
        update_cash(transaction_total)
        
        # 3. Nettoyage si quantité nulle
        if df.at[idx, 'Quantité'] <= 0:
             pass # On garde la ligne pour l'historique ou on pourrait la supprimer
             
        return True, f"✅ Vente validée : +{transaction_total:.2f}€ ajoutés aux liquidités."
        
    return False, "Erreur inconnue"

# --- 4. BARRE LATÉRALE : CENTRE D'OPÉRATIONS ---
with st.sidebar:
    st.title("Opérations")
    st.caption("Passer un ordre ou faire un virement")
    
    with st.form("order_form"):
        op_type = st.selectbox("Type d'opération", ["Achat", "Vente", "Apport Cash (Virement)"])
        
        # Si c'est un apport, on simplifie l'affichage
        if op_type == "Apport Cash (Virement)":
            ticker_input = "CASH"
            qty_input = 0.0
            price_input = st.number_input("Montant du virement (€)", min_value=1.0, step=100.0)
            st.info(f"Ceci ajoutera {price_input}€ à vos liquidités.")
        else:
            # Liste des tickers existants + option d'en taper un nouveau
            existing_tickers = [t for t in st.session_state['portfolio_df']['Ticker'].unique() if t != "CASH"]
            ticker_select = st.selectbox("Actif", existing_tickers + ["NOUVEAU..."])
            
            if ticker_select == "NOUVEAU...":
                ticker_input = st.text_input("Symbole (ex: AI.PA)", value="AI.PA").upper()
            else:
                ticker_input = ticker_select
                
            c1, c2 = st.columns(2)
            qty_input = c1.number_input("Quantité", min_value=0.01, step=1.0)
            price_input = c2.number_input("Prix Unitaire (€)", min_value=0.01, step=0.1)
            
            total_prev = qty_input * price_input
            st.write(f"Total opération : **{total_prev:,.2f} €**")

        submitted = st.form_submit_button("Valider l'Ordre")
        
        if submitted:
            if op_type == "Apport Cash (Virement)":
                update_cash(price_input)
                st.success(f"💰 Virement de {price_input}€ reçu !")
            else:
                success, msg = execute_order(op_type, ticker_input, qty_input, price_input)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    st.markdown("---")
    st.warning("⚠️ Note : Les modifications sont temporaires (Session).")

# --- 5. RÉCUPÉRATION PRIX ET CALCULS ---

@st.cache_data(ttl=60) # Cache court pour réactivité
def get_market_data(tickers):
    prices = {"CASH": 1.0}
    valid_tickers = [t for t in tickers if t != "CASH"]
    if valid_tickers:
        try:
            data = yf.download(valid_tickers, period="1d", progress=False)['Close']
            if hasattr(data, 'columns') and len(valid_tickers) > 1:
                last_row = data.iloc[-1]
                for t in valid_tickers:
                    prices[t] = float(last_row[t]) if t in last_row else 0.0
            elif not data.empty:
                 prices[valid_tickers[0]] = float(data.iloc[-1])
        except: pass
    return prices

# On travaille sur le DataFrame de la Session
df = st.session_state['portfolio_df'].copy()
current_prices = get_market_data(df['Ticker'].unique())

df['Prix_Actuel'] = df['Ticker'].apply(lambda x: current_prices.get(x, df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['Plus_Value'] = df['Valo'] - df['Investi']
df['Perf_%'] = df.apply(lambda x: ((x['Prix_Actuel'] - x['PRU']) / x['PRU'] * 100) if x['PRU'] > 0 else 0, axis=1)

# Agrégats
cash_dispo = df[df['Ticker']=="CASH"]['Valo'].sum()
investi_titres = df[df['Ticker']!="CASH"]['Investi'].sum()
valo_titres = df[df['Ticker']!="CASH"]['Valo'].sum()
total_pf = valo_titres + cash_dispo
total_pv = total_pf - (investi_titres + cash_dispo) # Calcul simplifié PV Latente

# --- 6. INTERFACE GRAPHIQUE ---

st.title("Terminal de Gestion")
st.caption(f"État du portefeuille en temps réel - {datetime.now().strftime('%H:%M')}")

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Portefeuille Total", f"{total_pf:,.2f} €")
k2.metric("Liquidités Disponibles", f"{cash_dispo:,.2f} €", help="Cash prêt à investir")
k3.metric("Montant Investi (Titres)", f"{investi_titres:,.2f} €")
k4.metric("Plus-Value Latente", f"{total_pv:+,.2f} €", f"{(total_pv/investi_titres)*100 if investi_titres>0 else 0:+.2f}%")

st.markdown("---")

tab_pf, tab_alloc = st.tabs(["📋 Portefeuille & Opérations", "📊 Répartition"])

with tab_pf:
    st.subheader("Positions en cours")
    st.dataframe(
        df[['Nom', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo', 'Perf_%']],
        column_config={
            "PRU": st.column_config.NumberColumn("PRU (Moyen)", format="%.2f €"),
            "Prix_Actuel": st.column_config.NumberColumn("Cours", format="%.2f €"),
            "Valo": st.column_config.NumberColumn("Valo", format="%.2f €"),
            "Perf_%": st.column_config.ProgressColumn("Perf.", format="%+.2f %%", min_value=-30, max_value=30)
        },
        hide_index=True, use_container_width=True
    )

with tab_alloc:
    fig = px.pie(df[df['Ticker']!="CASH"], values='Valo', names='Nom', hole=0.5)
    st.plotly_chart(fig, use_container_width=True)