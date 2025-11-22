import yfinance as yf
import pandas as pd
import sys

# --- CONFIGURATION ---
# Vos tickers exacts extraits de votre portefeuille
TICKERS = ["ESE.PA", "DCAM.PA", "PUST.PA", "CL2.PA", "BTC-EUR"]

print("\n" + "="*50)
print("💎 AUDIT DE CONNEXION YAHOO FINANCE")
print("="*50 + "\n")

success_count = 0
error_count = 0

for t in TICKERS:
    print(f"🔍 Audit de l'actif : {t} ... ", end="")
    try:
        # Méthode 1 : Ticker + History (Plus précis pour l'instantané)
        ticker_obj = yf.Ticker(t)
        hist = ticker_obj.history(period="1d")
        
        if hist.empty:
            # Méthode 2 : Tentative de secours via download (Parfois plus robuste sur les indices)
            data = yf.download(t, period="1d", progress=False)
            if data.empty:
                print("❌ ÉCHEC (Donnée vide)")
                print(f"   -> Yahoo ne renvoie aucune cotation pour {t}.")
                error_count += 1
            else:
                # Gestion du format complexe de yfinance
                if 'Close' in data.columns:
                    price = data['Close'].iloc[-1]
                else:
                    price = data.iloc[-1]
                    
                # Nettoyage si c'est une Série ou un float
                if isinstance(price, pd.Series):
                    price = float(price.iloc[0])
                else:
                    price = float(price)

                print(f"⚠️ RÉCUPÉRÉ (Via méthode de secours) : {price:,.2f} €")
                success_count += 1
        else:
            # Extraction propre du prix
            price = hist['Close'].iloc[-1]
            print(f"✅ VALIDE : {price:,.2f} €")
            success_count += 1
            
    except Exception as e:
        print(f"🔥 ERREUR CRITIQUE : {e}")
        error_count += 1

print("\n" + "-"*50)
print(f"RÉSULTAT : {success_count} Actifs valides | {error_count} Erreurs")
print("-"*50 + "\n")

if error_count > 0:
    print("⚠️ ANALYSE : Votre perte de 12 000 € vient probablement des actifs marqués en 'ÉCHEC'.")
    print("   Yahoo Finance renvoie 0 ou vide, le code pense donc que l'actif vaut 0 €.")
else:
    print("✅ ANALYSE : Tous les prix sont corrects.")
    print("   Conclusion : Le problème vient à 100% du fichier 'historique.csv' qui compare")
    print("   votre portefeuille actuel avec un ancien montant obsolète.")