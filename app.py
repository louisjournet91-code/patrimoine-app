import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURATION & STYLE BANQUE PRIVÉE ---
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
    div[data-testid="stForm"] { border: 1px solid #d4af37; background-color: #fffdf5; padding: 10px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# --- 2. INITIALISATION DES DONNÉES (MÉMOIRE SESSION) ---

INITIAL_PORTFOLIO = {
    "Ticker": ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR", "CASH"],
    "Nom": ["BNP S&P 500", "Amundi World", "Lyxor Nasdaq", "Amundi USA x2", "Bitcoin", "Liquidités"],
    "Type": ["ETF Action", "ETF Action", "ETF Tech", "ETF Levier", "Crypto", "Cash"],
    "Quantité": [141.0, 716.0, 55.0, 176.0, 0.01, 510.84],
    "PRU": [24.41, 4.68, 71.73, 19.71, 90165.46, 1.00]
}

# Si c'est le premier chargement, on crée la "base de données" temporaire
if 'portfolio_df' not in st.session_state:
    st.session_state['portfolio_df'] = pd.DataFrame(INITIAL_PORTFOLIO)

# --- 3. LOGIQUE DES OPÉRATIONS (MOTEUR TRANSACTIONNEL) ---

def update_cash(amount):
    """Ajoute (ou retire si négatif) de l'argent à la ligne CASH de façon robuste."""
    df = st.session_state['portfolio_df']
    # On utilise un masque booléen pour trouver la ligne CASH à coup sûr
    mask = df['Ticker'] == "CASH"
    
    if mask.any():
        # On récupère l'ancienne valeur
        current_val = df.loc[mask, 'Quantité'].values[0]
        # On met à jour
        df.loc[mask, 'Quantité'] = current_val + amount
        st.session_state['portfolio_df'] = df # Sauvegarde forcée
        return True
    return False

def execute_order(action, ticker, qty, price):
    """Exécute Achat ou Vente avec recalcul du PRU"""
    df = st.session_state['portfolio_df']
    
    # Vérifier si l'actif existe, sinon l'ajouter
    if ticker not in df['Ticker'].values:
        new_row = {"Ticker": ticker, "Nom": ticker, "Type": "Autre", "Quantité": 0.0, "PRU": 0.0}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state['portfolio_df'] = df # Sauvegarde intermédiaire

    # Localisation de la ligne de l'actif
    mask = df['Ticker'] == ticker
    idx = df.index[mask].tolist()[0]
    
    current_qty = df.at[idx, 'Quantité']
    current_pru = df.at[idx, 'PRU']
    transaction_total = qty * price
    
    if action == "Achat":
        # Vérif Cash
        cash_dispo = df.loc[df['Ticker'] == "CASH", 'Quantité'].values[0]
        if cash_dispo < transaction_total:
            return False, "❌ Liquidités insuffisantes !"
            
        # Calcul Nouveau PRU (Moyenne Pondérée)
        new_qty = current_qty + qty
        new_pru = ((current_qty * current_pru) + (qty * price)) / new_qty
        
        # Mise à jour Actif
        df.at[idx, 'Quantité'] = new_qty
        df.at[idx, 'PRU'] = new_pru
        st.session_state['portfolio_df'] = df # Sauvegarde
        
        # Débit Cash
        update_cash(-transaction_total)
        return True, f"✅ Achat validé : {qty} {ticker}. Nouveau PRU : {new_pru:.2f}€"

    elif action == "Vente":
        if current_qty < qty:
            return False, "❌ Quantité insuffisante !"
            
        # Mise à jour Actif (PRU ne change pas à la vente)
        df.at[idx, 'Quantité'] = current_qty - qty
        st.session_state['portfolio_df'] = df # Sauvegarde
        
        # Crédit Cash
        update_cash(transaction_total)
        return True, f"✅ Vente validée : +{transaction_total:.2f}€ sur le compte."
        
    return False, "Erreur technique."

# --- 4. BARRE LATÉRALE (OPERATIONS) ---
with st.sidebar:
    st.header("🏦 Opérations")
    
    with st.form("ops_form"):
        type_op = st.radio("Action", ["Apport Cash", "Achat Titre", "Vente Titre"], horizontal=True)
        
        # Affichage dynamique selon le choix
        if type_op == "Apport Cash":
            st.info("Virement vers le compte Liquidités")
            ticker_in = "CASH"
            qty_in = 0.0
            amount_in = st.number_input("Montant (€)", min_value=10.0, step=50.0)
        else:
            # Liste des actifs existants pour faciliter la sélection
            tickers_list = [t for t in st.session_state['portfolio_df']['Ticker'].unique() if t != "CASH"]
            ticker_in = st.selectbox("Actif", tickers_list + ["NOUVEAU..."])
            if ticker_in == "NOUVEAU...":
                ticker_in = st.text_input("Symbole (ex: MC.PA)").upper()
                
            c1, c2 = st.columns(2)
            qty_in = c1.number_input("Quantité", min_value=0.01, step=1.0)
            price_in = c2.number_input("Prix Unitaire (€)", min_value=0.01, step=0.1)
            amount_in = 0 # Pas utilisé ici, on utilise qty * price

        btn = st.form_submit_button("Exécuter l'ordre", type="primary")
        
        if btn:
            if type_op == "Apport Cash":
                update_cash(amount_in)
                st.success(f"💰 +{amount_in}€ ajoutés aux liquidités !")
                st.rerun() # Force le rafraîchissement immédiat de la page
            else:
                ok, msg = execute_order("Achat" if type_op=="Achat Titre" else "Vente", ticker_in, qty_in, price_in)
                if ok:
                    st.success(msg)
                    st.rerun() # Force le rafraîchissement
                else:
                    st.error(msg)

# --- 5. CALCULS & DATA (LIVE) ---

@st.cache_data(ttl=60)
def get_prices(tickers):
    """Récupère les prix actuels et veille"""
    real_tickers = [t for t in tickers if t != "CASH"]
    prices = {"CASH": {"cur": 1.0, "prev": 1.0}}
    
    if real_tickers:
        try:
            # On prend 5 jours pour être sûr d'avoir la veille
            data = yf.download(real_tickers, period="5d", progress=False)['Close']
            
            # Gestion retour unique ou multiple
            if len(real_tickers) == 1:
                # Series
                prices[real_tickers[0]] = {"cur": float(data.iloc[-1]), "prev": float(data.iloc[-2])}
            else:
                # DataFrame
                last = data.iloc[-1]
                prev = data.iloc[-2]
                for t in real_tickers:
                    if t in last.index:
                        prices[t] = {"cur": float(last[t]), "prev": float(prev[t])}
        except: pass
    return prices

# Récupération du DataFrame Session
df = st.session_state['portfolio_df'].copy()
market_data = get_prices(df['Ticker'].unique())

# Application des prix
df['Prix_Actuel'] = df['Ticker'].apply(lambda x: market_data.get(x, {}).get("cur", df.loc[df['Ticker']==x, 'PRU'].values[0]))
df['Prix_Veille'] = df['Ticker'].apply(lambda x: market_data.get(x, {}).get("prev", df.loc[df['Ticker']==x, 'PRU'].values[0]))

# Calculs financiers
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['Plus_Value'] = df['Valo'] - df['Investi']
df['Perf_%'] = df.apply(lambda x: ((x['Prix_Actuel'] - x['PRU']) / x['PRU'] * 100) if x['PRU']>0 else 0, axis=1)
df['Var_Jour_€'] = df['Valo'] - (df['Quantité'] * df['Prix_Veille'])

# Totaux
cash = df[df['Ticker']=="CASH"]['Valo'].sum()
investi_titres = df[df['Ticker']!="CASH"]['Investi'].sum()
valo_titres = df[df['Ticker']!="CASH"]['Valo'].sum()
total_pf = valo_titres + cash
total_pv = valo_titres - investi_titres

# --- 6. INTERFACE PRINCIPALE ---

st.title("Terminal de Gestion")
st.caption(f"Date valeur : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Portefeuille Total", f"{total_pf:,.2f} €")
k2.metric("Liquidités", f"{cash:,.2f} €", help="Disponible pour investir")
k3.metric("Montant Investi (Titres)", f"{investi_titres:,.2f} €")
k4.metric("PV Latente", f"{total_pv:+,.2f} €", f"{(total_pv/investi_titres)*100 if investi_titres>0 else 0:+.2f}%")

st.markdown("---")

# ONGLETS (AVEC LE RETOUR DE LA PROJECTION)
tab1, tab2, tab3 = st.tabs(["📋 Portefeuille", "📊 Analyse", "🔮 Projection Rente"])

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
        # Waterfall
        fig_w = go.Figure(go.Waterfall(
            orientation="v", measure=["relative"] * len(df),
            x=df['Nom'], y=df['Plus_Value'],
            connector={"line":{"color":"#cbd5e1"}},
            decreasing={"marker":{"color":"#ef4444"}}, increasing={"marker":{"color":"#10b981"}}
        ))
        fig_w.update_layout(title="Contribution PV (€)", template="simple_white")
        st.plotly_chart(fig_w, use_container_width=True)
    with c2:
        # Donut
        fig_d = px.pie(df[df['Ticker']!="CASH"], values='Valo', names='Nom', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_d.update_layout(title="Répartition Actifs", showlegend=False, template="simple_white")
        fig_d.add_annotation(text=f"{valo_titres/1000:.1f}k€", showarrow=False, font=dict(size=20))
        st.plotly_chart(fig_d, use_container_width=True)

with tab3:
    st.header("Simulateur d'Indépendance Financière")
    
    col_in, col_out = st.columns([1, 2])
    
    with col_in:
        st.markdown("#### Paramètres")
        # On prend le total actuel comme point de départ
        capital_depart = total_pf
        apport_mensuel = st.number_input("Apport mensuel (€)", 0, 5000, 500, step=100)
        rendement_annuel = st.slider("Rendement annuel (%)", 2.0, 15.0, 8.0, 0.5)
        duree_ans = st.slider("Horizon (années)", 5, 35, 15)
        
    with col_out:
        # Calcul Intérêts Composés
        data_proj = []
        capital = capital_depart
        
        for annee in range(1, duree_ans + 1):
            # Intérêts gagnés cette année
            interets = capital * (rendement_annuel / 100)
            # Ajout des versements (12 mois)
            versements = apport_mensuel * 12
            # Nouveau capital
            capital = capital + interets + versements
            
            data_proj.append({
                "Année": datetime.now().year + annee,
                "Capital": capital,
                "Intérêts Cumulés": interets # Juste pour l'année en cours ici, simplifie
            })
            
        df_proj = pd.DataFrame(data_proj)
        
        # Affichage Graphique Area Chart (Effet Premium)
        fig_proj = px.area(df_proj, x="Année", y="Capital", title="Évolution de votre Patrimoine", color_discrete_sequence=["#0f172a"])
        fig_proj.update_layout(template="simple_white")
        st.plotly_chart(fig_proj, use_container_width=True)
        
        capital_final = df_proj.iloc[-1]['Capital']
        rente_mensuelle_4pct = (capital_final * 0.04) / 12
        
        st.success(f"🎯 **Objectif {datetime.now().year + duree_ans}** : **{capital_final:,.0f} €**")
        st.info(f"💸 Rente passive potentielle (Règle des 4%) : **{rente_mensuelle_4pct:,.0f} € / mois**")