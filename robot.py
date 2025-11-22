import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import warnings

# --- 1. CONFIGURATION ---
warnings.simplefilter(action='ignore', category=FutureWarning)

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# Liste officielle des colonnes pour l'historique
HIST_COLS = [
    "Date", "Total", "PEA", "BTC", "Plus-value", "Delta", "PV du Jour", 
    "ESE", "Flux (€)", "PF_Return_TWR", "ESE_Return", 
    "PF_Index100", "ESE_Index100", "PF_Index100.1", "ESE_Index100.1"
]

print(f"\n--- 💎 ROBOT ULTIMATE ESTATE : {datetime.now().strftime('%d/%m/%Y %H:%M')} ---")

# --- 2. LECTURE DU PORTEFEUILLE ---
if not os.path.exists(FILE_PORTFOLIO):
    print(f"❌ ERREUR : Le fichier {FILE_PORTFOLIO} est introuvable.")
    exit()

try:
    # Lecture robuste (virgule ou point-virgule)
    df = pd.read_csv(FILE_PORTFOLIO, sep=None, engine='python', dtype={'Quantité': str, 'PRU': str})
    
    # Nettoyage des chiffres (virgules -> points)
    def clean_float(x):
        if pd.isna(x): return 0.0
        return float(str(x).replace(',', '.').replace(' ', '').replace('€', '').replace('%', ''))

    df['Quantité'] = df['Quantité'].apply(clean_float)
    df['PRU'] = df['PRU'].apply(clean_float)
    
    print(f"✅ Portefeuille chargé : {len(df)} lignes détectées.")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE lecture portefeuille : {e}")
    exit()

# --- 3. RECUPERATION DES PRIX (Méthode Validée par Diagnostic) ---
tickers_list = df['Ticker'].unique().tolist()
real_tickers = [t for t in tickers_list if t != "CASH"]
prices = {"CASH": 1.0}

print(f"📡 Connexion Yahoo Finance pour {len(real_tickers)} actifs...")

for t in real_tickers:
    try:
        # On utilise la méthode qui a fonctionné dans le diagnostic
        tick_obj = yf.Ticker(t)
        hist = tick_obj.history(period="1d")
        
        if not hist.empty:
            prices[t] = float(hist['Close'].iloc[-1])
        else:
            # Fallback download
            data = yf.download(t, period="1d", progress=False)
            if not data.empty:
                val = data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1]
                prices[t] = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
            else:
                print(f"⚠️ Pas de prix pour {t}, utilisation du PRU.")
                prices[t] = 0.0 # Sera remplacé par PRU
    except:
        print(f"⚠️ Erreur API sur {t}")

# Application des prix
df['Prix_Actuel'] = df.apply(lambda x: prices.get(x['Ticker'], 0.0), axis=1)
# Si prix 0 (échec API), on prend le PRU pour ne pas casser la valo
df['Prix_Actuel'] = df.apply(lambda x: x['PRU'] if x['Prix_Actuel'] == 0.0 and x['Ticker'] != "CASH" else x['Prix_Actuel'], axis=1)

# --- 4. CALCULS ---
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
total_pf = df['Valo'].sum()
val_btc = df[df['Ticker'].str.contains("BTC", na=False)]['Valo'].sum()
val_pea = total_pf - val_btc
total_pv = total_pf - (df['Quantité'] * df['PRU']).sum()

# Prix de référence S&P 500 (pour comparaison)
ese_price = prices.get("ESE.PA", 0.0)

print(f"💰 VALORISATION TOTALE : {total_pf:,.2f} €")

# --- 5. ENREGISTREMENT HISTORIQUE ---
today_str = datetime.now().strftime("%d/%m/%Y")

# Chargement ou Création
if os.path.exists(FILE_HISTORY):
    try:
        df_hist = pd.read_csv(FILE_HISTORY, sep=';')
    except:
        df_hist = pd.DataFrame(columns=HIST_COLS)
else:
    print("✨ Création d'un nouvel historique vierge.")
    df_hist = pd.DataFrame(columns=HIST_COLS)

# Vérif doublon jour
if not df_hist.empty and today_str in df_hist['Date'].astype(str).values:
    print(f"ℹ️ Déjà enregistré pour {today_str}. Suppression de l'ancienne entrée du jour...")
    df_hist = df_hist[df_hist['Date'] != today_str]

# Calcul Variation (Delta)
if not df_hist.empty:
    last_row = df_hist.iloc[-1]
    prev_total = float(str(last_row['Total']).replace(',', '.'))
    prev_ese = float(str(last_row['ESE']).replace(',', '.'))
    
    delta = total_pf - prev_total
    
    # Indices Base 100 (TWR simplifié)
    last_idx_pf = float(str(last_row.get('PF_Index100', 100)).replace(',', '.'))
    perf_pct = (delta / prev_total) if prev_total != 0 else 0
    new_idx_pf = last_idx_pf * (1 + perf_pct)
    
    last_idx_ese = float(str(last_row.get('ESE_Index100', 100)).replace(',', '.'))
    ese_perf = (ese_price - prev_ese)/prev_ese if prev_ese != 0 else 0
    new_idx_ese = last_idx_ese * (1 + ese_perf)
    
else:
    # Premier lancement (Base 100)
    delta = 0.0
    perf_pct = 0.0
    ese_perf = 0.0
    new_idx_pf = 100.0
    new_idx_ese = 100.0
    print("🏁 Initialisation de la Base 100.")

# Nouvelle Ligne
new_row = {
    "Date": today_str,
    "Total": round(total_pf, 2),
    "PEA": round(val_pea, 2),
    "BTC": round(val_btc, 2),
    "Plus-value": round(total_pv, 2),
    "Delta": round(delta, 2),
    "PV du Jour": round(delta, 2), # Simplifié hors flux
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
# Forcer l'ordre des colonnes
df_final = df_final.reindex(columns=HIST_COLS)

df_final.to_csv(FILE_HISTORY, sep=';', index=False, encoding='utf-8-sig')
print(f"✅ SUCCÈS : Historique mis à jour dans {FILE_HISTORY}")