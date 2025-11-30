import streamlit as st
import pandas as pd
import yfinance as yf
import os

# --- CONSTANTES ---
FILE_PORTFOLIO = 'portefeuille.csv'
FILE_HISTORY = 'historique.csv'

# --- FONCTIONS DE DONNÉES ---

def load_data():
    """Charge les données avec sécurité."""
    # 1. Portefeuille
    if os.path.exists(FILE_PORTFOLIO):
        try:
            df_pf = pd.read_csv(FILE_PORTFOLIO, sep=',', dtype=str)
        except Exception as e:
            st.error(f"Erreur lecture portefeuille: {e}")
            df_pf = pd.DataFrame()
    else:
        df_pf = pd.DataFrame()

    # Nettoyage
    def clean_float(x):
        if pd.isna(x): return 0.0
        return float(str(x).replace(',', '.').replace('€', '').replace(' ', '').replace('%', ''))

    if not df_pf.empty:
        for c in ['Quantité', 'PRU']:
            if c in df_pf.columns: df_pf[c] = df_pf[c].apply(clean_float)

    # 2. Historique
    df_h = pd.DataFrame()
    if os.path.exists(FILE_HISTORY):
        try:
            df_h = pd.read_csv(FILE_HISTORY, sep=';', on_bad_lines='skip', engine='python')
            df_h['Date'] = pd.to_datetime(df_h['Date'], dayfirst=True, errors='coerce')
            df_h = df_h.dropna(subset=['Date']).sort_values('Date')
            
            for col in ['Total', 'PF_Index100', 'ESE_Index100']:
                if col in df_h.columns:
                    df_h[col] = df_h[col].apply(clean_float)
        except Exception as e:
            st.error(f"Erreur lecture historique: {e}")
            
    return df_pf, df_h

@st.cache_data(ttl=300)
def get_live_prices(tickers):
    """
    Récupère les prix.
    Stratégie : Tente le téléchargement de groupe (Rapide).
    En cas d'échec, bascule sur la méthode unitaire (Fiable & Testée).
    """
    prices = {"CASH": {"cur": 1.0, "prev": 1.0}}
    real_ticks = [t for t in tickers if t != "CASH" and isinstance(t, str)]
    
    if not real_ticks:
        return prices

    # --- MÉTHODE 1 : GROUPE (Rapide) ---
    success_bulk = False
    try:
        # On laisse yfinance gérer la session (curl_cffi fera le travail en coulisse)
        data = yf.download(tickers=real_ticks, period="5d", progress=False, group_by='ticker', threads=True)
        
        # Si on reçoit des données valides, on traite
        if not data.empty:
            for t in real_ticks:
                try:
                    # Gestion du format MultiIndex ou Simple de yfinance
                    df_t = data[t] if len(real_ticks) > 1 and t in data else data
                    
                    if not df_t.empty and 'Close' in df_t.columns:
                        vals = df_t['Close'].dropna()
                        if not vals.empty:
                            cur = float(vals.iloc[-1])
                            prev = float(vals.iloc[-2]) if len(vals) > 1 else cur
                            prices[t] = {"cur": cur, "prev": prev}
                            success_bulk = True
                except: pass
    except Exception as e:
        print(f"⚠️ Bulk download failed, switching to surgical mode: {e}")

    # --- MÉTHODE 2 : CHIRURGICALE (La méthode qui a marché dans votre test) ---
    # On ne lance ceci que pour les tickers qui ont échoué ou si le bulk a planté
    for t in real_ticks:
        if t not in prices or prices[t]['cur'] == 0.0:
            try:
                # C'est exactement la commande que vous avez tapée dans le terminal
                hist = yf.Ticker(t).history(period="5d")
                if not hist.empty:
                    cur = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else cur
                    prices[t] = {"cur": cur, "prev": prev}
            except Exception:
                # Si vraiment tout échoue (ex: ticker invalide)
                prices[t] = {"cur": 0.0, "prev": 0.0}

    return prices

@st.cache_data(ttl=3600)
def get_market_indices():
    """Récupère les indices de marché."""
    targets = {"S&P 500": "^GSPC", "CAC 40": "^FCHI", "Bitcoin": "BTC-EUR", "VIX": "^VIX"}
    res = []
    
    for name, tick in targets.items():
        try:
            h = yf.Ticker(tick).history(period="5d")
            if not h.empty:
                cur = float(h['Close'].iloc[-1])
                prev = float(h['Close'].iloc[-2]) if len(h)>1 else cur
                perf = ((cur-prev)/prev)*100 if prev != 0 else 0
                res.append({"Indice": name, "Prix": cur, "24h %": perf})
        except: pass
    return pd.DataFrame(res)

# --- FONCTIONS UI (BENTO) ---
def create_bento_card(asset, card_bg, border_color, text_color, metric_gradient):
    color_perf = "#10b981" if asset['Perf_%'] >= 0 else "#ef4444"
    bg_perf = "rgba(16, 185, 129, 0.15)" if asset['Perf_%'] >= 0 else "rgba(239, 68, 68, 0.15)"
    arrow = "▲" if asset['Perf_%'] >= 0 else "▼"
    
    return f"""
    <div style="background-color: {card_bg}; border: 1px solid {border_color}; border-radius: 20px; padding: 20px; margin-bottom: 20px; transition: transform 0.2s;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-weight: 700; font-size: 1.1rem; color: {text_color}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 60%;">{asset['Nom']}</span>
            <span style="background-color: {bg_perf}; color: {color_perf}; padding: 4px 10px; border-radius: 10px; font-size: 0.85rem; font-weight: 600;">{arrow} {asset['Perf_%']:+.2f}%</span>
        </div>
        <div style="margin-bottom: 15px;">
            <div style="font-size: 0.85rem; opacity: 0.6; color: {text_color};">Valorisation</div>
            <div style="font-size: 1.8rem; font-weight: 800; background: {metric_gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; color: {text_color};">{asset['Valo']:,.2f} €</div>
        </div>
        <div style="display: flex; justify-content: space-between; border-top: 1px solid {border_color}; padding-top: 12px; font-size: 0.9rem; color: {text_color};">
            <div style="display: flex; flex-direction: column;"><span style="opacity: 0.5; font-size: 0.75rem;">Quantité</span><span style="font-weight: 500;">{asset['Quantité']:.4f}</span></div>
            <div style="display: flex; flex-direction: column; text-align: right;"><span style="opacity: 0.5; font-size: 0.75rem;">Prix Actuel</span><span style="font-weight: 500;">{asset['Prix_Actuel']:.2f} €</span></div>
        </div>
    </div>
    """