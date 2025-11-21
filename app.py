import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURATION & STYLE PREMIUM ---
st.set_page_config(page_title="Gestion Patrimoniale", layout="wide", page_icon="🏛️")

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

# --- 2. INITIALISATION DES DONNÉES ---

INITIAL_PORTFOLIO = {
    "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
    "Nom": ["BNP S&P 500", "Amundi World", "Lyxor Nasdaq", "Amundi USA x2", "Bitcoin", "Liquidités"],
    "Type": ["ETF Action", "ETF Action", "ETF Tech", "ETF Levier", "Crypto", "Cash"],
    "Quantité": [141.0, 716.0, 55.0, 176.0, 0.01, 510.84],
    "PRU": [24.41, 4.68, 71.73, 19.71, 90165.46, 1.00]
}

if 'portfolio_df' not in st.session_state:
    df_init = pd.DataFrame(INITIAL_PORTFOLIO)
    # On force le typage float pour éviter les bugs de calcul
    df_init['Quantité'] = df_init['Quantité'].astype(float)
    df_init['PRU'] = df_init['PRU'].astype(float)
    st.session_state['portfolio_df'] = df_init

# --- 3. MOTEUR TRANSACTIONNEL (SEPARÉ) ---

def operation_tresorerie(amount):
    """Gère uniquement le Cash (Apport/Retrait)"""
    df = st.session_state['portfolio_df']
    # Utilisation de .loc avec masque pour être infaillible
    mask = df['Ticker'] == 'CASH'
    
    if mask.any():
        current_cash = df.loc[mask, 'Quantité'].values[0]
        df.loc[mask, 'Quantité'] = current_cash + amount
        st.session_state['portfolio_df'] = df
        return True
    return False

def operation_trading(action, ticker, qty, price, nom_actif="Nouvel Actif", type_actif="Action"):
    """Gère uniquement l'Achat/Vente de titres"""
    df = st.session_state['portfolio_df']
    
    # 1. Gestion Actif Inconnu (Création)
    if ticker not in df['Ticker'].values:
        if action == "Vente": return False, "Impossible de vendre un actif que vous ne possédez pas."
        new_row = pd.DataFrame([{
            "Ticker": ticker, "Nom": nom_actif, "Type": type_actif, 
            "Quantité": 0.0, "PRU": 0.0
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    
    # 2. Récupération Cash Dispo
    cash_mask = df['Ticker'] == 'CASH'
    cash_dispo = df.loc[cash_mask, 'Quantité'].values[0]
    total_ordre = qty * price
    
    # 3. Ciblage de la ligne Actif
    asset_mask = df['Ticker'] == ticker
    current_qty = df.loc[asset_mask, 'Quantité'].values[0]
    current_pru = df.loc[asset_mask, 'PRU'].values[0]
    
    if action == "Achat":
        if cash_dispo < total_ordre:
            return False, f"Liquidités insuffisantes (Manque {total_ordre - cash_dispo:.2f}€)"
        
        # Calcul PRU Pondéré
        new_qty = current_qty + qty
        new_pru = ((current_qty * current_pru) + (qty * price)) / new_qty
        
        # Application
        df.loc[asset_mask, 'Quantité'] = new_qty
        df.loc[asset_mask, 'PRU'] = new_pru
        df.loc[cash_mask, 'Quantité'] = cash_dispo - total_ordre
        
        st.session_state['portfolio_df'] = df
        return True, f"✅ Achat exécuté : {qty} {ticker} (Nouveau PRU: {new_pru:.2f}€)"
        
    elif action == "Vente":
        if current_qty < qty:
            return False, "Vous vendez plus que vous ne possédez !"
            
        # Application (Le PRU ne change pas à la vente)
        df.loc[asset_mask, 'Quantité'] = current_qty - qty
        df.loc[cash_mask, 'Quantité'] = cash_dispo + total_ordre
        
        st.session_state['portfolio_df'] = df
        return True, f"✅ Vente exécutée. +{total_ordre:.2f}€ crédités."
        
    return False, "Erreur inconnue"

# --- 4. SIDEBAR : LES DEUX GUICHETS DISTINCTS ---

with st.sidebar:
    st.header("Opérations")
    
    # --- MODULE 1 : TRÉSORERIE ---
    with st.expander("💰 Trésorerie (Virements)", expanded=True):
        st.caption("Alimenter le compte Liquidités")
        montant_virement = st.number_input("Montant (€)", min_value=0.0, step=100.0, key="input_virement")
        
        if st.button("Valider le Virement", type="secondary", use_container_width=True):
            if montant_virement > 0:
                operation_tresorerie(montant_virement)
                st.success(f"+{montant_virement}€ ajoutés !")
                st.rerun()

    st.markdown("---")

    # --- MODULE 2 : TRADING ---
    with st.expander("📈 Trading (Achat/Vente)", expanded=True):
        st.caption("Passer un ordre de bourse")
        
        sens = st.radio("Sens", ["Achat", "Vente"], horizontal=True)
        
        # Sélection Actif (Existant ou Nouveau)
        existing_tickers = [t for t in st.session_state['portfolio_df']['Ticker'].unique() if t != "CASH"]
        mode_actif = st.radio("Actif", ["Existant", "Nouveau"], horizontal=True, label_visibility="collapsed")
        
        if mode_actif == "Existant":
            ticker = st.selectbox("Sélectionner", existing_tickers)
            nom_actif = "" # Pas besoin
        else:
            ticker = st.text_input("Ticker (ex: AI.PA)").upper()
            nom_actif = st.text_input("Nom (ex: Air Liquide)")
        
        c1, c2 = st.columns(2)
        qty = c1.number_input("Quantité", min_value=0.01, step=1.0)
        price = c2.number_input("Prix Limite", min_value=0.01, step=0.1)
        
        st.markdown(f"**Total Ordre : {qty*price:,.2f} €**")
        
        if st.button(f"Confirmer {sens}", type="primary", use_container_width=True):
            success, msg = operation_trading(sens, ticker, qty, price, nom_actif)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

# --- 5. CALCULS & AFFICHAGE (FRONTEND) ---

@st.cache_data(ttl=60)
def get_prices(tickers):
    prices = {"CASH": {"cur": 1.0, "prev": 1.0}}
    real_tickers = [t for t in tickers if t != "CASH"]
    if real_tickers:
        try:
            data = yf.download(real_tickers, period="5d", progress=False)['Close']
            # Gestion Single vs Multi Index
            if len(real_tickers) == 1:
                prices[real_tickers[0]] = {"cur": float(data.iloc[-1]), "prev": float(data.iloc[-2])}
            else:
                last = data.iloc[-1]
                prev = data.iloc[-2]
                for t in real_tickers:
                    if t in last.index:
                        prices[t] = {"cur": float(last[t]), "prev": float(prev[t])}
        except: pass
    return prices

# Récupération Data Session
df = st.session_state['portfolio_df'].copy()
market_data = get_prices(df['Ticker'].unique())

# Injection des prix
df['Prix_Actuel'] = df['Ticker'].apply(lambda x: market_data.get(x, {}).get("cur", df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Prix_Veille'] = df['Ticker'].apply(lambda x: market_data.get(x, {}).get("prev", df.loc[df['Ticker']==x, 'PRU'].values[0]))

# Calculs Financiers
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['Plus_Value'] = df['Valo'] - df['Investi']
df['Perf_%'] = df.apply(lambda x: ((x['Prix_Actuel'] - x['PRU']) / x['PRU'] * 100) if x['PRU']>0 else 0, axis=1)
df['Var_Jour_€'] = df['Valo'] - (df['Quantité'] * df['Prix_Veille'])

# Agrégats
cash = df[df['Ticker']=="CASH"]['Valo'].sum()
investi_titres = df[df['Ticker']!="CASH"]['Investi'].sum()
valo_titres = df[df['Ticker']!="CASH"]['Valo'].sum()
total_pf = valo_titres + cash
total_pv = valo_titres - investi_titres

# --- 6. INTERFACE ---

st.title("Terminal de Gestion")
st.caption(f"Dernière valorisation : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Portefeuille Total", f"{total_pf:,.2f} €")
k2.metric("Liquidités", f"{cash:,.2f} €")
k3.metric("Investi (Titres)", f"{investi_titres:,.2f} €")
k4.metric("PV Latente", f"{total_pv:+,.2f} €", f"{(total_pv/investi_titres)*100 if investi_titres>0 else 0:+.2f}%")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📋 Positions", "📊 Analyse", "🔮 Projection"])

with tab1:
    st.dataframe(
        df[['Nom', 'Quantité', 'PRU', 'Prix_Actuel', 'Valo', 'Var_Jour_€', 'Perf_%']],
        column_config={
            "PRU": st.column_config.NumberColumn("PRU", format="%.2f €"),
            "Prix_Actuel": st.column_config.NumberColumn("Cours", format="%.2f €"),
            "Valo": st.column_config.NumberColumn("Valo", format="%.2f €"),
            "Var_Jour_€": st.column_config.NumberColumn("Var. Jour", format="%+.2f €"),
            "Perf_%": st.column_config.ProgressColumn("Perf.", format="%+.2f %%", min_value=-30, max_value=30)
        },
        hide_index=True, use_container_width=True
    )

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig_w = go.Figure(go.Waterfall(
            orientation="v", measure=["relative"] * len(df),
            x=df['Nom'], y=df['Plus_Value'],
            connector={"line":{"color":"#cbd5e1"}},
            decreasing={"marker":{"color":"#ef4444"}}, increasing={"marker":{"color":"#10b981"}}
        ))
        fig_w.update_layout(title="Contribution PV (€)", template="simple_white")
        st.plotly_chart(fig_w, use_container_width=True)
    with c2:
        fig_d = px.pie(df[df['Ticker']!="CASH"], values='Valo', names='Nom', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_d.update_layout(title="Répartition", showlegend=False, template="simple_white")
        st.plotly_chart(fig_d, use_container_width=True)

with tab3:
    st.header("Projection")
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown("#### Paramètres")
        start_cap = total_pf
        monthly_add = st.number_input("Apport mensuel (€)", 0, 5000, 500, step=100)
        rate = st.slider("Rendement (%)", 2.0, 15.0, 8.0, 0.5)
        years = st.slider("Horizon (ans)", 5, 35, 15)
    with col_out:
        proj_data = []
        cap = start_cap
        for y in range(1, years + 1):
            interest = cap * (rate / 100)
            cap = cap + interest + (monthly_add * 12)
            proj_data.append({"Année": datetime.now().year + y, "Capital": cap})
        
        df_proj = pd.DataFrame(proj_data)
        fig_p = px.area(df_proj, x="Année", y="Capital", title="Patrimoine Futur", color_discrete_sequence=["#0f172a"])
        fig_p.update_layout(template="simple_white")
        st.plotly_chart(fig_p, use_container_width=True)
        st.success(f"🎯 Capital à terme : {cap:,.0f} €")