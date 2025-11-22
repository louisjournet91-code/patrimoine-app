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

# --- 2. LECTURE DU PORTEFEUILLE ---
try:
    df = pd.read_csv(FILE_PORTFOLIO, sep=',', encoding='utf-8', dtype={'Quantité': str, 'PRU': str})
    df.columns = df.columns.str.strip()
    
    def clean_float(x):
        if pd.isna(x): return 0.0
        return float(str(x).replace(',', '.').replace(' ', '').replace('€', '').replace('%', ''))

    df['Quantité'] = df['Quantité'].apply(clean_float)
    df['PRU'] = df['PRU'].apply(clean_float)
    print(f"✅ Portefeuille chargé : {len(df)} lignes.")

except Exception as e:
    print(f"❌ ERREUR FATALE lecture portefeuille : {e}")
    exit()

# --- 3. RECUPERATION ROBUSTE DES PRIX ---
real_tickers = [t for t in df['Ticker'].unique() if t != "CASH"]
prices = {"CASH": 1.0}

print(f"📡 Connexion Yahoo Finance pour {len(real_tickers)} actifs...")

for t in real_tickers:
    try:
        # On cherche sur 1 mois pour éviter les trous de cotation (fériés, etc.)
        tick_obj = yf.Ticker(t)
        hist = tick_obj.history(period="1mo")
        
        if not hist.empty:
            prices[t] = float(hist['Close'].iloc[-1])
            print(f"   ✅ {t} : {prices[t]:.2f} €")
        else:
            # Tentative de secours
            data = yf.download(t, period="5d", progress=False)
            if not data.empty:
                vals = data['Close']
                val = vals.iloc[-1] if hasattr(vals, 'iloc') else vals
                prices[t] = float(val)
                print(f"   ⚠️ {t} (Download) : {prices[t]:.2f} €")
            else:
                print(f"   ❌ PRIX INTROUVABLE pour {t}. Utilisation PRU par sécurité.")
                prices[t] = 0.0 
    except Exception as e:
        print(f"   ❌ Erreur API sur {t}: {e}")
        prices[t] = 0.0

# Application des prix (Fallback PRU si 0.0)
df['Prix_Actuel'] = df.apply(lambda x: prices.get(x['Ticker'], 0.0), axis=1)
# Si prix = 0, on prend le PRU pour ne pas casser la valorisation totale
df['Prix_Actuel'] = df.apply(lambda x: x['PRU'] if x['Prix_Actuel'] <= 0 and x['Ticker'] != "CASH" else x['Prix_Actuel'], axis=1)

# --- 4. CALCULS DE RICHESSE ---
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
total_pf = df['Valo'].sum()
val_btc = df[df['Ticker'].str.contains("BTC", na=False)]['Valo'].sum()
val_pea = total_pf - val_btc
total_pv = total_pf - (df['Quantité'] * df['PRU']).sum()

ese_price = prices.get("ESE.PA", 0.0)
if ese_price == 0: ese_price = df.loc[df['Ticker']=="ESE.PA", "PRU"].values[0] if not df.loc[df['Ticker']=="ESE.PA"].empty else 0

print(f"💰 VALORISATION TOTALE : {total_pf:,.2f} €")

# --- 5. SAUVEGARDE HISTORIQUE ---
today_str = datetime.now().strftime("%d/%m/%Y")

if os.path.exists(FILE_HISTORY):
    try:
        df_hist = pd.read_csv(FILE_HISTORY, sep=';')
    except:
        df_hist = pd.DataFrame(columns=HIST_COLS)
else:
    df_hist = pd.DataFrame(columns=HIST_COLS)

# Suppression doublon du jour
if not df_hist.empty:
    df_hist = df_hist[df_hist['Date'] != today_str]

# Calculs Variation vs J-1
delta = 0.0
perf_pct = 0.0
ese_perf = 0.0
new_idx_pf = 100.0
new_idx_ese = 100.0

if not df_hist.empty:
    last_row = df_hist.iloc[-1]
    try:
        prev_total = float(str(last_row['Total']).replace(',', '.'))
        prev_ese = float(str(last_row['ESE']).replace(',', '.'))
        
        if prev_total > 0:
            delta = total_pf - prev_total
            perf_pct = (delta / prev_total)
            last_idx_pf = float(str(last_row.get('PF_Index100', 100)).replace(',', '.'))
            new_idx_pf = last_idx_pf * (1 + perf_pct)
        
        if prev_ese > 0:
            ese_perf = (ese_price - prev_ese) / prev_ese
            last_idx_ese = float(str(last_row.get('ESE_Index100', 100)).replace(',', '.'))
            new_idx_ese = last_idx_ese * (1 + ese_perf)
            
    except Exception as e:
        print(f"⚠️ Erreur calcul indices: {e}")

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

df_final = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
df_final.to_csv(FILE_HISTORY, sep=';', index=False, encoding='utf-8-sig')
print(f"✅ SUCCÈS : Patrimoine sauvegardé.")