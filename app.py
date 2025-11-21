import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- 1. CONFIGURATION ET STYLE PREMIUM ---
st.set_page_config(page_title="Gestion Privée", layout="wide")

# Initialisation de la "Mémoire" du site (Session State)
# C'est ici que vos données vivent tant que le site est ouvert
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=['Date', 'Type', 'Actif', 'Quantité', 'Prix', 'Total'])
if 'cash_balance' not in st.session_state:
    st.session_state.cash_balance = 0.0

# --- 2. LE MOTEUR DE CALCUL (BACKEND) ---
def get_live_price(ticker):
    """Récupère le vrai prix du marché. Si échec, retourne 0."""
    try:
        if ticker == "CASH": return 1.0
        # Astuce : yfinance a besoin de suffixes (ex: AIR.PA pour Paris)
        # Ici on simplifie pour l'exemple
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if not history.empty:
            return history['Close'].iloc[-1]
        return 0.0
    except:
        return 0.0

def ajouter_transaction(date, type_op, actif, qte, prix):
    """Enregistre une opération dans le grand livre"""
    total = qte * prix
    
    # Impact sur le Cash
    if type_op == "Versement Espèces":
        st.session_state.cash_balance += total
        actif = "LIQUIDITÉS"
    elif type_op == "Achat Titre":
        st.session_state.cash_balance -= total
    elif type_op == "Vente Titre":
        st.session_state.cash_balance += total

    # Ajout au journal
    new_row = {
        'Date': date, 'Type': type_op, 'Actif': actif, 
        'Quantité': qte, 'Prix': prix, 'Total': total
    }
    st.session_state.transactions = pd.concat([st.session_state.transactions, pd.DataFrame([new_row])], ignore_index=True)

# --- 3. L'INTERFACE DE SAISIE (SIDEBAR) ---
with st.sidebar:
    st.header("📝 Saisir une Opération")
    
    type_operation = st.selectbox("Type d'opération", ["Versement Espèces", "Achat Titre", "Vente Titre"])
    
    date_op = st.date_input("Date", datetime.now())
    
    if type_operation == "Versement Espèces":
        actif_input = "CASH"
        qte_input = 1.0
        prix_input = st.number_input("Montant du versement (€)", min_value=0.0, step=100.0)
    else:
        actif_input = st.text_input("Symbole Actif (ex: AIR.PA, EPA:ESE)", value="EPA:ESE")
        qte_input = st.number_input("Quantité", min_value=0.0, step=1.0)
        prix_input = st.number_input("Prix Unitaire (€)", min_value=0.0, step=0.1)
        
        # Petit calculateur d'aide
        st.caption(f"Total de l'ordre : {qte_input * prix_input:,.2f} €")

    if st.button("Valider l'opération", type="primary"):
        ajouter_transaction(date_op, type_operation, actif_input, qte_input, prix_input)
        st.success("Opération enregistrée !")

# --- 4. LE TABLEAU DE BORD (FRONTEND) ---
st.title("🏛️ Votre Patrimoine en Temps Réel")

# Calculs des Positions Actuelles (Agrégation)
df = st.session_state.transactions
if not df.empty:
    # On filtre pour ne garder que les achats/ventes de titres
    mouvements_titres = df[df['Type'].isin(['Achat Titre', 'Vente Titre'])]
    
    if not mouvements_titres.empty:
        # Calcul du PRU et des quantités par actif
        # Note: C'est une version simplifiée (Moyenne pondérée)
        portfolio = mouvements_titres.groupby('Actif').agg(
            Quantité_Totale=('Quantité', 'sum'),
            Investi_Total=('Total', 'sum')
        ).reset_index()
        
        # Calcul du PRU
        portfolio['PRU'] = portfolio['Investi_Total'] / portfolio['Quantité_Totale']
        
        # Récupération des prix actuels (Simulation pour la démo si pas de connexion)
        # Dans la version finale, on active la ligne get_live_price
        portfolio['Prix_Actuel'] = portfolio['Actif'].apply(lambda x: 28.64 if 'ESE' in x else (110.5 if 'AIR' in x else get_live_price(x))) 
        
        # Calculs finaux
        portfolio['Valorisation'] = portfolio['Quantité_Totale'] * portfolio['Prix_Actuel']
        portfolio['Plus-Value €'] = portfolio['Valorisation'] - portfolio['Investi_Total']
        portfolio['Performance %'] = (portfolio['Plus-Value €'] / portfolio['Investi_Total']) * 100
        
        # Totaux Généraux
        total_investi = portfolio['Valorisation'].sum()
        total_cash = st.session_state.cash_balance
        patrimoine_total = total_investi + total_cash
        perf_globale = portfolio['Plus-Value €'].sum()
        
        # --- AFFICHAGE DES KPIs ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patrimoine Net", f"{patrimoine_total:,.2f} €")
        col2.metric("Dont Liquidités", f"{total_cash:,.2f} €", f"{(total_cash/patrimoine_total)*100:.1f}%")
        col3.metric("Investi en Titres", f"{total_investi:,.2f} €")
        col4.metric("Plus-Value Latente", f"{perf_globale:,.2f} €", f"{(perf_globale/total_investi if total_investi>0 else 0)*100:.2f} %")
        
        st.divider()
        
        # --- AFFICHAGE DU TABLEAU DÉTAILLÉ ---
        st.subheader("Détail du Portefeuille")
        
        # Mise en forme du tableau pour faire "Pro"
        st.dataframe(
            portfolio.style.format({
                "PRU": "{:.2f} €",
                "Prix_Actuel": "{:.2f} €",
                "Valorisation": "{:.2f} €",
                "Plus-Value €": "{:+.2f} €",
                "Performance %": "{:+.2f} %"
            }),
            use_container_width=True
        )
        
    else:
        st.info("Aucun titre en portefeuille. Utilisez le menu de gauche pour acheter des actifs.")
        st.metric("Liquidités Disponibles", f"{st.session_state.cash_balance:,.2f} €")

else:
    st.warning("👋 Bienvenue Monsieur. Commencez par saisir un 'Versement Espèces' dans le menu de gauche pour alimenter votre compte.")

# --- 5. HISTORIQUE DES TRANSACTIONS (BAS DE PAGE) ---
with st.expander("Voir l'Historique des Opérations"):
    st.dataframe(st.session_state.transactions, use_container_width=True)