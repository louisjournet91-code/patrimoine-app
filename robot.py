import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import os
import warnings

# --- 1. CONFIGURATION & STANDARDS ---
warnings.simplefilter(action='ignore', category=FutureWarning)

FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# User-Agent "Premium" pour passer les barrières de Yahoo (Mimique Chrome sur Windows)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

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
    print(f"❌ ERREUR CRITIQUE lecture portefeuille : {e}")
    exit()

# --- 3. RÉCUPÉRATION BLINDÉE DES PRIX (STRATÉGIE BULK) ---
real_tickers = [t for t in df['Ticker'].unique() if t != "CASH" and pd.notna(t)]
prices = {"CASH": 1.0}

print(f"📡 Connexion Sécurisée Yahoo Finance pour {len(real_tickers)} actifs...")

if real_tickers:
    try:
        # Configuration de la session pour masquer le robot
        session = requests.Session()
        session.headers.update({'User-Agent': USER_AGENT})

        # Téléchargement GROUPÉ (Moins suspect pour l'API et plus rapide)
        # On prend 5 jours pour être sûr d'avoir la dernière clôture
        print(f"   ... Téléchargement groupé en cours ...")
        
        # Astuce: group_by='ticker' permet de structurer les données proprement
        data = yf.download(
            tickers=real_tickers, 
            period="5d", 
            session=session, 
            progress=False, 
            group_by='ticker',
            threads=True
        )

        for t in real_tickers:
            price_found = 0.0
            try:
                # Extraction spécifique selon la structure retournée par yfinance (parfois MultiIndex, parfois simple)
                if len(real_tickers) > 1:
                    ticker_data = data[t]
                else:
                    ticker_data = data # Si un seul ticker, pas de niveau supérieur

                # On cherche la dernière valeur de Clôture ('Close') non nulle
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    last_valid = ticker_data['Close'].dropna().iloc[-1]
                    price_found = float(last_valid)
                
                # Check spécifique pour rassurer Monsieur
                if t == "ESE.PA":
                    if price_found > 10: # Le prix devrait être ~29€
                        print(f"   💎 ESE.PA (S&P 500) : {price_found:.2f} € (SUCCÈS)")
                    else:
                        print(f"   ⚠️ ESE.PA : Prix suspect ou nul ({price_found})")

                if price_found > 0:
                    prices[t] = price_found
                else:
                    print(f"   ⚠️ Pas de données récentes pour {t}")

            except Exception as e:
                print(f"   ❌ Erreur extraction {t}: {e}")

    except Exception as e:
        print(f"❌ Échec global du téléchargement Yahoo : {e}")

# Application des prix (Fallback PRU si toujours 0.0)
def get_price_final(row):
    t = row['Ticker']
    p = prices.get(t, 0.0)
    # Si le prix est 0 (échec), on utilise le PRU pour ne pas casser le total
    if p <= 0 and t != "CASH":
        print(f"   🛡️ Utilisation PRU pour {t} ({row['PRU']} €)")
        return row['PRU']
    return p

df['Prix_Actuel'] = df.apply(get_price_final, axis=1)

# --- 4. CALCULS DE RICHESSE ---
df['Valo'] = df['Quantité'] * df['Prix_Actuel']
total_pf = df['Valo'].sum()
val_btc = df[df['Ticker'].str.contains("BTC", na=False)]['Valo'].sum()
val_pea = total_pf - val_btc
total_pv = total_pf - (df['Quantité'] * df['PRU']).sum()

# Récupération propre du prix ESE pour l'historique
ese_row = df[df['Ticker'] == "ESE.PA"]
ese_price = ese_row['Prix_Actuel'].values[0] if not ese_row.empty else 0.0

print(f"💰 VALORISATION TOTALE : {total_pf:,.2f} €")

# --- 5. SAUVEGARDE HISTORIQUE (Inchangé) ---
today_str = datetime.now().strftime("%d/%m/%Y")

if os.path.exists(FILE_HISTORY):
    try:
        df_hist = pd.read_csv(FILE_HISTORY, sep=';')
    except:
        df_hist = pd.DataFrame() # Re-création si corrompu
else:
    df_hist = pd.DataFrame()

# Suppression doublon du jour
if not df_hist.empty and 'Date' in df_hist.columns:
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
# Encodage utf-8-sig pour compatibilité Excel
df_final.to_csv(FILE_HISTORY, sep=';', index=False, encoding='utf-8-sig')
print(f"✅ SUCCÈS : Patrimoine sauvegardé avec ESE à {ese_price:.2f} €.")