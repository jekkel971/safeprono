import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------
# ⚙️ Clé API (optionnelle, pour test local)
# ---------------------------
API_KEY = "TA_CLE_API_ICI"  # ⚠️ Remplace par ta clé si tu veux tester avec l'API, sinon laisse vide

# ---------------------------
# 🔹 Championnats
# ---------------------------
CHAMPIONNATS = ["🇫🇷 Ligue 1", "🏴 Premier League", "🇪🇸 La Liga"]

# ---------------------------
# 🔹 Générer des matchs simulés pour le week-end
# ---------------------------
def generate_weekend_matches():
    weekend_start = datetime.now()
    matches = []

    for champ in CHAMPIONNATS:
        for i in range(5):  # 5 matchs simulés par championnat
            home = f"Team_Home_{i+1}"
            away = f"Team_Away_{i+1}"
            match_date = weekend_start + timedelta(days=np.random.randint(0,7), hours=np.random.randint(12,22))

            # Cotes simulées réalistes autour de 1.5
            diff_strength = np.random.uniform(-0.5,0.5)
            cote_home = round(1.5 - 0.05*diff_strength,2)
            cote_away = round(1.5 + 0.05*diff_strength,2)

            matches.append({
                "Championnat": champ,
                "Match": f"{home} vs {away}",
                "home_team": home,
                "away_team": away,
                "cote_home": cote_home,
                "cote_away": cote_away,
                "Date": match_date
            })
    
    df = pd.DataFrame(matches)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M")  # pour affichage Streamlit
    return df

# ---------------------------
# 🔹 Calcul Score Sécurité et Winner
# ---------------------------
def calculate_scores(df):
    df["Winner"] = np.where(df["cote_home"] < df["cote_away"], df["home_team"], df["away_team"])
    df["Score_Sécurité"] = (1 - abs(df["cote_home"] - df["cote_away"])) * 100
    return df

# ---------------------------
# 🔹 Streamlit Interface
# ---------------------------
st.set_page_config(page_title="Matchs Safe du Week-end", layout="wide")
st.title("⚽ Matchs Safe du Week-end (Gratuit)")
st.caption("Simulation avec cotes réalistes pour test local et classement des matchs les plus sûrs")

# ✅ Bouton avec session_state pour garder l'état
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

if st.button("Lancer l'analyse 🧠"):
    st.session_state.run_analysis = True

if st.session_state.run_analysis:
    # Générer les matchs
    df_matches = generate_weekend_matches()
    st.write(f"📊 Matchs simulés générés : {df_matches.shape[0]}")
    
    # Calculer scores et winner
    df_matches = calculate_scores(df_matches)

    # Top 3–4 matchs les plus sûrs
    top = df_matches.sort_values(by="Score_Sécurité", ascending=False).head(4)
    st.success("🏆 Les 3–4 matchs les plus sûrs du week-end :")
    st.dataframe(top[["Championnat","Match","Winner","Score_Sécurité","Date"]], use_container_width=True)

    # Télécharger CSV complet
    st.download_button(
        "📥 Télécharger tous les résultats (CSV)",
        df_matches.to_csv(index=False).encode("utf-8"),
        "matchs_safe_local.csv",
        "text/csv"
    )
