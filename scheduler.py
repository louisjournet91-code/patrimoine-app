import pandas as pd
import yfinance as yf
from datetime import datetime
import os

# FICHIERS
FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# --- FONCTIONS UTILITAIRES ---
def safe_float(x):
    if pd.isna(x) or x == "": return 0.0
    if isinstance(x, (float, int)): return float(x)
    if isinstance(x, str):
        x = x.replace(' ', '').replace('%', '').replace('€', '').replace(',', '.')
        if x.count('.') > 1: x = x.replace('.', '', x.count('.') - 1)
        try: return float(x)
        except: return 0.0
    return 0.0

def get_prices(tickers):
    prices = {"CASH": 1.0}
    real = [t for t in tickers if t != "CASH"]
    if real:
        try:
            d = yf.download(real, period="5d", progress=False)['Close']
            if len(real) == 1:
                prices[real[0]] = float(d.iloc[-1])
            else:
                l = d.iloc[-1]
                for t in real:
                    if t in l.index: prices[t] = float(l[t])
        except: pass
    return prices

# --- EXÉCUTION ---
print("--- Démarrage du Robot ---")

# 1. Charger le Portefeuille
if os.path.exists(FILE_PORTFOLIO):
    try:
        df = pd.read_csv(FILE_PORTFOLIO, sep=';')
        if df.shape[1] < 2: df = pd.read_csv(FILE_PORTFOLIO, sep=',')
        
        df['Quantité'] = df['Quantité'].apply(safe_float)
        df['PRU'] = df['PRU'].apply(safe_float)
    except Exception as e:
        print(f"Erreur lecture portefeuille: {e}")
        exit()
else:
    print("Pas de fichier portefeuille !")
    exit()

# 2. Récupérer les Prix Live
print("Récupération des prix...")
market = get_prices(df['Ticker'].unique())
df['Prix_Actuel'] = df['Ticker'].apply(lambda x: market.get(x, 0.0))

# 3. Calculs
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
df['Investi'] = df['Quantité'] * df['PRU']
df['PV'] = df['Valo'] - df['Investi']

# Totaux
total_pf = df['Valo'].sum()
val_btc = df[df['Ticker'].str.contains("BTC")]['Valo'].sum()
val_pea = df[~df['Ticker'].str.contains("BTC")]['Valo'].sum()
total_pv = df['PV'].sum()

# Prix ESE pour l'historique
try:
    ese_price = df.loc[df['Ticker'].str.contains("ESE"), 'Prix_Actuel'].values[0]
except:
    ese_price = 0.0

# 4. Charger l'historique pour comparaison (J-1)
if os.path.exists(FILE_HISTORY):
    try:
        df_hist = pd.read_csv(FILE_HISTORY, sep=';', engine='python')
    except:
        df_hist = pd.read_csv(FILE_HISTORY, sep=',', engine='python')
else:
    print("Création fichier historique")
    # Colonnes par défaut si fichier absent
    df_hist = pd.DataFrame(columns=["Date", "Total", "PEA", "BTC", "Plus-value", "Delta", "PV_du_Jour", 
    "ESE", "Flux_(€)", "PF_Return_TWR", "ESE_Return", "PF_Index100", "ESE_Index100"])

# Récupération Veille
if not df_hist.empty:
    last_row = df_hist.iloc[-1]
    prev_total = safe_float(last_row['Total'])
    prev_ese = safe_float(last_row['ESE'])
    prev_pf_idx = safe_float(last_row['PF_Index100'])
    prev_ese_idx = safe_float(last_row['ESE_Index100'])
    prev_pv = safe_float(last_row['Plus-value'])
else:
    prev_total = total_pf
    prev_ese = 0.0
    prev_pf_idx = 100.0
    prev_ese_idx = 100.0
    prev_pv = 0.0

# 5. Calculs Indicateurs (Vos Formules)
flux = 0.0 # Flux à 0 car c'est une maj automatique nuit
delta = total_pv - prev_pv
pv_jour = total_pf - prev_total - flux

denom = prev_total + flux
pf_return = (total_pf - prev_total - flux) / denom if denom != 0 else 0.0

ese_return = (ese_price - prev_ese) / prev_ese if prev_ese != 0 else 0.0

pf_index100 = prev_pf_idx * (1 + pf_return)
ese_index100 = prev_ese_idx * (1 + ese_return)

# 6. Sauvegarde
today = datetime.now().strftime("%d/%m/%Y")

if today not in df_hist['Date'].astype(str).values:
    new_data = {
        "Date": today,
        "Total": round(total_pf, 2),
        "PEA": round(val_pea, 2),
        "BTC": round(val_btc, 2),
        "Plus-value": round(total_pv, 2),
        "Delta": round(delta, 2),
        "PV_du_Jour": round(pv_jour, 2),
        "ESE": round(ese_price, 2),
        "Flux_(€)": 0,
        "PF_Return_TWR": round(pf_return, 4),
        "ESE_Return": round(ese_return, 4),
        "PF_Index100": round(pf_index100, 2),
        "ESE_Index100": round(ese_index100, 2),
        "PF_Index100.1": round(pf_index100 - 100, 2),
        "ESE_Index100.1": round(ese_index100 - 100, 2)
    }
    
    # Alignement colonnes
    new_row = pd.DataFrame([new_data])
    if not df_hist.empty:
        new_row = new_row.reindex(columns=df_hist.columns, fill_value=0)
        
    df_final = pd.concat([df_hist, new_row], ignore_index=True)
    
    # Écriture
    df_final.to_csv(FILE_HISTORY, index=False, sep=';')
    print(f"Succès ! Ligne ajoutée pour le {today}")
else:
    print("Déjà fait pour aujourd'hui.")