import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import warnings

# --- 1. CONFIGURATION ---
warnings.simplefilter(action='ignore', category=FutureWarning)

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

HIST_COLS = [
    "Date", "Total", "PEA", "BTC", "Plus-value", "Delta", "PV du Jour", 
    "ESE", "Flux (€)", "PF_Return_TWR", "ESE_Return", 
    "PF_Index100", "ESE_Index100", "PF_Index100.1", "ESE_Index100.1"
]

print(f"\n--- 💎 ROBOT ULTIMATE ESTATE : {datetime.now().strftime('%d/%m/%Y %H:%M')} ---")

# --- 2. AUTORÉPARATION DU PORTEFEUILLE ---
# Si le fichier est absent ou vide (0 octet), on le régénère immédiatement
if not os.path.exists(FILE_PORTFOLIO) or os.path.getsize(FILE_PORTFOLIO) == 0:
    print("⚠️ Fichier portefeuille vide ou manquant. Restauration des données cryptées...")
    raw_portfolio = """Ticker,Nom,Type,Quantité,PRU
ESE.PA,BNP S&P 500,ETF Action,141.0,24.41
DCAM.PA,Amundi World,ETF Action,716.0,4.68
PUST.PA,Lyxor Nasdaq,ETF Tech,55.0,71.73
CL2.PA,Amundi USA x2,ETF Levier,176.0,19.71
BTC-EUR,Bitcoin,Crypto,0.0142,90165.46
CASH,Liquidités,Cash,510.84,1.0"""
    with open(FILE_PORTFOLIO, "w", encoding="utf-8") as f:
        f.write(raw_portfolio)
    print("✅ Portefeuille restauré avec succès.")

# --- 3. LECTURE DU PORTEFEUILLE ---
try:
    # Lecture standard avec virgule
    df = pd.read_csv(FILE_PORTFOLIO, sep=',', encoding='utf-8', dtype={'Quantité': str, 'PRU': str})
    
    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()
    
    # Fonction de nettoyage numérique
    def clean_float(x):
        if pd.isna(x): return 0.0
        return float(str(x).replace(',', '.').replace(' ', '').replace('€', '').replace('%', ''))

    # Vérification colonne critique
    if 'Quantité' not in df.columns:
        # Fallback si encodage cassé
        cols = [c for c in df.columns if c.startswith('Quantit')]
        if cols:
            df.rename(columns={cols[0]: 'Quantité'}, inplace=True)
        else:
            raise KeyError("Colonne 'Quantité' introuvable")

    df['Quantité'] = df['Quantité'].apply(clean_float)
    df['PRU'] = df['PRU'].apply(clean_float)
    
    print(f"✅ Portefeuille chargé : {len(df)} lignes.")

except Exception as e:
    print(f"❌ ERREUR FATALE lecture portefeuille : {e}")
    exit()

# --- 4. RECUPERATION DES PRIX ---
tickers_list = df['Ticker'].unique().tolist()
real_tickers = [t for t in tickers_list if t != "CASH"]
prices = {"CASH": 1.0}

print(f"📡 Connexion Yahoo Finance pour {len(real_tickers)} actifs...")

for t in real_tickers:
    try:
        tick_obj = yf.Ticker(t)
        # history(period="1d") est plus robuste pour éviter les DataFrames vides
        hist = tick_obj.history(period="5d") # On prend 5j pour être sûr d'avoir une clôture
        
        if not hist.empty:
            prices[t] = float(hist['Close'].iloc[-1])
        else:
            # Tentative désespérée via download si history échoue
            data = yf.download(t, period="1d", progress=False)
            if not data.empty:
                val = data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1]
                prices[t] = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
            else:
                print(f"⚠️ Prix introuvable pour {t}, utilisation du PRU.")
                prices[t] = 0.0 
    except Exception as e:
        print(f"⚠️ Erreur API sur {t}: {e}")

# Application des prix (Fallback PRU si échec API pour éviter valeur 0)
df['Prix_Actuel'] = df.apply(lambda x: prices.get(x['Ticker'], 0.0), axis=1)
df['Prix_Actuel'] = df.apply(lambda x: x['PRU'] if x['Prix_Actuel'] == 0.0 and x['Ticker'] != "CASH" else x['Prix_Actuel'], axis=1)

# --- 5. CALCULS DE RICHESSE ---
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
total_pf = df['Valo'].sum()
val_btc = df[df['Ticker'].str.contains("BTC", na=False)]['Valo'].sum()
val_pea = total_pf - val_btc
total_pv = total_pf - (df['Quantité'] * df['PRU']).sum()

ese_price = prices.get("ESE.PA", 0.0)

print(f"💰 VALORISATION TOTALE : {total_pf:,.2f} €")

# --- 6. SAUVEGARDE HISTORIQUE ---
today_str = datetime.now().strftime("%d/%m/%Y")

# Chargement Historique existant
if os.path.exists(FILE_HISTORY):
    try:
        df_hist = pd.read_csv(FILE_HISTORY, sep=';')
        # Sécurisation: conversion des colonnes numériques
        for col in ['Total', 'ESE', 'PF_Index100', 'ESE_Index100']:
            if col in df_hist.columns:
                df_hist[col] = df_hist[col].apply(lambda x: str(x).replace(',', '.'))
    except:
        df_hist = pd.DataFrame(columns=HIST_COLS)
else:
    df_hist = pd.DataFrame(columns=HIST_COLS)

# Suppression doublon du jour si réexécution
if not df_hist.empty and today_str in df_hist['Date'].astype(str).values:
    print(f"ℹ️ Mise à jour de l'entrée existante pour {today_str}...")
    df_hist = df_hist[df_hist['Date'] != today_str]

# Calculs Variation vs J-1
delta = 0.0
perf_pct = 0.0
ese_perf = 0.0
new_idx_pf = 100.0
new_idx_ese = 100.0

if not df_hist.empty:
    # Récupération dernière ligne valide (avant aujourd'hui)
    last_row = df_hist.iloc[-1]
    try:
        prev_total = float(str(last_row['Total']).replace(',', '.'))
        prev_ese = float(str(last_row['ESE']).replace(',', '.'))
        
        delta = total_pf - prev_total
        
        # Indices Base 100
        last_idx_pf = float(str(last_row.get('PF_Index100', 100)).replace(',', '.'))
        perf_pct = (delta / prev_total) if prev_total != 0 else 0
        new_idx_pf = last_idx_pf * (1 + perf_pct)
        
        last_idx_ese = float(str(last_row.get('ESE_Index100', 100)).replace(',', '.'))
        ese_perf = (ese_price - prev_ese)/prev_ese if prev_ese != 0 else 0
        new_idx_ese = last_idx_ese * (1 + ese_perf)
    except Exception as e:
        print(f"⚠️ Erreur calcul variation: {e}. Reset indices.")

# Création ligne du jour
new_row = {
    "Date": today_str,
    "Total": round(total_pf, 2),
    "PEA": round(val_pea, 2),
    "BTC": round(val_btc, 2),
    "Plus-value": round(total_pv, 2),
    "Delta": round(delta, 2),
    "PV du Jour": round(delta, 2),
    "ESE": round(ese_price, 2),
    "Flux (€)": 0,
    "PF_Return_TWR": f"{perf_pct*100:.2f}".replace('.', ','),
    "ESE_Return": f"{ese_perf*100:.2f}".replace('.', ','),
    "PF_Index100": round(new_idx_pf, 2),
    "ESE_Index100": round(new_idx_ese, 2),
    "PF_Index100.1": round(new_idx_pf - 100, 2),
    "ESE_Index100.1": round(new_idx_ese - 100, 2)
}

# Ajout et Sauvegarde
df_final = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
df_final = df_final.reindex(columns=HIST_COLS)

# On sauvegarde en forçant le point-virgule pour le format français/Excel
df_final.to_csv(FILE_HISTORY, sep=';', index=False, encoding='utf-8-sig')
print(f"✅ SUCCÈS : Patrimoine du {today_str} enregistré.")