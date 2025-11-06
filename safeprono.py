import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------
# ⚙️ Clé API directement dans le code
# ---------------------------
API_KEY = "8b95c22ea5fe5a230b130b83e21a7535"  # ⚠️ Remplace par ta clé API

# ---------------------------
# 🔹 Championnats
# ---------------------------
CHAMPIONNATS = ["🇫🇷 Ligue 1", "🏴 Premier League", "🇪🇸 La Liga"]

# ---------------------------
# 🔹 Générer des matchs simulés pour le week-end
# ---------------------------
def generate_weekend_matches():
    weekend_start = datetime.now()
    weekend_end = weekend_start + timedelta(days=7)
    matches = []

    for champ in CHAMPIONNATS:
        for i in range(5):  # 5 matchs simulés par championnat
            home = f"Team_Home_{i+1}"
            away = f"Team_Away_{i+1}"
            match_date = weekend_start + timedelta(days=np.random.randint(0,7), hours=np.random.randint(12,22))

            # Générer cotes réalistes autour de 1.5
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
    return pd.DataFrame(matches)
