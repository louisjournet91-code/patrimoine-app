import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import warnings

# Ignorer les avertissements techniques non critiques
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# Liste EXACTE des colonnes pour s'aligner avec le Streamlit (avec espaces)
HIST_COLS = [
    "Date", "Total", "PEA", "BTC", "Plus-value", "Delta", "PV du Jour", 
    "ESE", "Flux (€)", "PF_Return_TWR", "ESE_Return", 
    "PF_Index100", "ESE_Index100", "PF_Index100.1", "ESE_Index100.1"
]

# --- FONCTIONS ROBUSTES ---
def safe_float(x):
    """Conversion blindée pour les formats français et pourcentages"""
    if pd.isna(x) or x == "": return 0.0
    s = str(x).strip().replace('"', '').replace('%', '').replace('€', '').replace(' ', '')
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        return 0.0

def get_prices_robust(tickers):
    """Récupération Yahoo Finance optimisée pour éviter les erreurs de format"""
    prices = {"CASH": 1.0}
    real_tickers = [t for t in tickers if t != "CASH" and isinstance(t, str)]
    
    if not real_tickers:
        return prices
        
    print(f"📡 Interrogation des marchés pour : {len(real_tickers)} actifs...")
    try:
        # Téléchargement groupé
        data = yf.download(real_tickers, period="5d", progress=False)
        
        # Gestion de la structure complexe de yfinance (Close peut être MultiIndex)
        if 'Close' in data.columns:
            closes = data['Close']
        else:
            closes = data # Cas rare ou fallback
            
        # Extraction du dernier prix disponible (iloc[-1])
        if len(real_tickers) == 1:
            # Si un seul ticker, closes est souvent une Series, pas un DataFrame
            last_price = float(closes.iloc[-1]) if hasattr(closes, 'iloc') else float(closes)
            prices[real_tickers[0]] = last_price
        else:
            # Si plusieurs tickers, c'est un DataFrame
            last_row = closes.iloc[-1]
            for t in real_tickers:
                if t in last_row.index:
                    prices[t] = float(last_row[t])
                    
    except Exception as e:
        print(f"⚠️ Avertissement API : {e}")
        
    return prices

# --- MAIN ---
if __name__ == "__main__":
    print(f"\n--- 💎 ROBOT ULTIMATE ESTATE : {datetime.now().strftime('%d/%m/%Y %H:%M')} ---")

    # 1. CHARGEMENT PORTEFEUILLE
    if not os.path.exists(FILE_PORTFOLIO):
        print("❌ Erreur : Fichier portefeuille introuvable.")
        exit()

    try:
        # Tentative lecture avec séparateur ; (Excel FR) puis , (Standard)
        df = pd.read_csv(FILE_PORTFOLIO, sep=';', dtype=str)
        if df.shape[1] < 2:
            df = pd.read_csv(FILE_PORTFOLIO, sep=',', dtype=str)
        
        # Nettoyage
        df['Quantité'] = df['Quantité'].apply(safe_float)
        df['PRU'] = df['PRU'].apply(safe_float)
    except Exception as e:
        print(f"❌ Erreur lecture fichier : {e}")
        exit()

    # 2. MISES À JOUR PRIX
    market_prices = get_prices_robust(df['Ticker'].unique())
    
    # Application des prix (Fallback sur PRU si prix non trouvé pour ne pas mettre 0)
    df['Prix_Actuel'] = df.apply(
        lambda x: market_prices.get(x['Ticker'], x['PRU'] if x['PRU'] > 0 else 0.0), 
        axis=1
    )

    # 3. CALCULS VALORISATION
    df['Valo'] = df['Quantité'] * df['Prix_Actuel']
    df['PV'] = df['Valo'] - (df['Quantité'] * df['PRU'])

    total_pf = df['Valo'].sum()
    val_btc = df[df['Ticker'].str.contains("BTC", na=False)]['Valo'].sum()
    val_pea = total_pf - val_btc # Tout ce qui n'est pas BTC est considéré PEA/Cash
    total_pv_latente = df['PV'].sum()

    # Prix ESE (Benchmark)
    try:
        ese_row = df[df['Ticker'].str.contains("ESE", na=False)]
        ese_price = ese_row['Prix_Actuel'].values[0] if not ese_row.empty else 0.0
    except: ese_price = 0.0

    print(f"💰 Valorisation calculée : {total_pf:,.2f} €")

    # 4. GESTION HISTORIQUE (Mode Append Sécurisé)
    today_str = datetime.now().strftime("%d/%m/%Y")
    
    # Chargement ou Création Historique
    if os.path.exists(FILE_HISTORY):
        try:
            df_hist = pd.read_csv(FILE_HISTORY, sep=';', on_bad_lines='skip')
            # Fallback virgule si échec lecture point-virgule
            if df_hist.shape[1] < 5:
                df_hist = pd.read_csv(FILE_HISTORY, sep=',', on_bad_lines='skip')
        except:
            df_hist = pd.DataFrame(columns=HIST_COLS)
    else:
        df_hist = pd.DataFrame(columns=HIST_COLS)

    # Vérification doublon journalier
    existing_dates = []
    if not df_hist.empty and 'Date' in df_hist.columns:
        # On normalise les dates en string pour comparaison
        existing_dates = df_hist['Date'].astype(str).tolist()

    if today_str in existing_dates:
        print(f"ℹ️ Mise à jour déjà effectuée pour le {today_str}. Arrêt.")
        exit()

    # 5. CALCULS PERFORMANCE (TWR)
    # Récupération J-1
    if not df_hist.empty:
        last = df_hist.iloc[-1]
        prev_total = safe_float(last.get('Total', total_pf))
        prev_ese = safe_float(last.get('ESE', ese_price))
        prev_pf_idx = safe_float(last.get('PF_Index100', 100))
        prev_ese_idx = safe_float(last.get('ESE_Index100', 100))
    else:
        prev_total = total_pf
        prev_ese = ese_price
        prev_pf_idx = 100.0
        prev_ese_idx = 100.0

    # Logique Flux : On suppose 0 pour un run automatique
    # (Pour être précis, il faudrait comparer le 'Investi' total J vs J-1)
    flux = 0.0 
    
    delta = total_pf - prev_total
    pv_jour = delta - flux # Gain net du jour
    
    # Calcul Rentabilité
    denom = prev_total + flux
    pf_ret = pv_jour / denom if denom != 0 else 0.0
    ese_ret = (ese_price - prev_ese) / prev_ese if prev_ese != 0 else 0.0

    # Calcul Indices Base 100
    pf_idx = prev_pf_idx * (1 + pf_ret)
    ese_idx = prev_ese_idx * (1 + ese_ret)

    # 6. ENREGISTREMENT
    new_data = {
        "Date": today_str,
        "Total": round(total_pf, 2),
        "PEA": round(val_pea, 2),
        "BTC": round(val_btc, 2),
        "Plus-value": round(total_pv_latente, 2),
        "Delta": round(delta, 2),        # Variation brute du jour
        "PV du Jour": round(pv_jour, 2), # Variation nette (hors flux)
        "ESE": round(ese_price, 2),
        "Flux (€)": 0,
        "PF_Return_TWR": f"{pf_ret*100:.2f}%".replace('.', ','),
        "ESE_Return": f"{ese_ret*100:.2f}%".replace('.', ','),
        "PF_Index100": round(pf_idx, 2),
        "ESE_Index100": round(ese_idx, 2),
        "PF_Index100.1": round(pf_idx - 100, 2),   # Pour affichage graphique simplifié
        "ESE_Index100.1": round(ese_idx - 100, 2)
    }

    # Création DataFrame ligne
    df_new = pd.DataFrame([new_data])
    
    # Alignement colonnes (Si l'historique a moins de colonnes que prévu)
    # On concatène pour ajouter la ligne
    df_final = pd.concat([df_hist, df_new], ignore_index=True)
    
    # Sécurisation : On force l'ordre des colonnes et on remplit les NaN
    columns_order = [c for c in HIST_COLS if c in df_final.columns]
    df_final = df_final[columns_order]

    # SAUVEGARDE FINALE
    # On force le point-virgule pour compatibilité maximale Excel/Streamlit
    try:
        df_final.to_csv(FILE_HISTORY, index=False, sep=';', encoding='utf-8-sig')
        print(f"✅ SUCCÈS : Données enregistrées pour le {today_str}")
        print(f"   Delta Jour : {delta:+.2f} €")
        print(f"   Index PF   : {pf_idx:.2f}")
    except Exception as e:
        print(f"❌ Erreur critique sauvegarde : {e}")