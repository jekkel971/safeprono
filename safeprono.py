import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ---------------------------
# ⚙️ Configuration API
# ---------------------------
API_KEY = os.getenv("ODDS_API_KEY")  # clé API de the-odds-api.com
BASE_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"  # exemple Premier League

# Championnats et endpoints correspondants
CHAMPIONNATS = {
    "🇫🇷 Ligue 1": "soccer_fra_ligue_one",
    "🏴 Premier League": "soccer_eng_premier_league",
    "🇪🇸 La Liga": "soccer_spain_la_liga"
}

MARKETS = "h2h"  # 1X2
ODDS_FORMAT = "decimal"

# ---------------------------
# 🔹 Récupérer les matchs du week-end
# ---------------------------
def get_upcoming_matches(sport_key):
    # Récupérer les prochains 20 matchs
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={API_KEY}&regions=eu&markets={MARKETS}&oddsFormat={ODDS_FORMAT}&dateFormat=iso"
    res = requests.get(url)
    if res.status_code != 200:
        st.warning(f"Impossible de récupérer les matchs pour {sport_key}. Status: {res.status_code}")
        return pd.DataFrame()
    
    data = res.json()
    weekend_start = datetime.now()
    weekend_end = weekend_start + timedelta(days=7)
    matches = []

    for m in data:
        try:
            match_time = datetime.fromisoformat(m["commence_time"].replace("Z","+00:00"))
            if not (weekend_start <= match_time <= weekend_end):
                continue

            home = m["home_team"]
            away = m["away_team"]

            # 🔹 Prendre la cote du bookmaker "best" (le premier disponible)
            odds = None
            for bookmaker in m["bookmakers"]:
                if "h2h" in bookmaker["markets"][0]["outcomes"][0]:
                    odds_list = bookmaker["markets"][0]["outcomes"]
                    odds_dict = {o["name"]: o["price"] for o in odds_list}
                    odds = odds_dict
                    break
            if odds is None:
                # utiliser la première disponible si aucune h2h trouvée
                odds = {o["name"]: o["price"] for o in m["bookmakers"][0]["markets"][0]["outcomes"]}

            cote_home = odds.get(home, np.nan)
            cote_away = odds.get(away, np.nan)
            if np.isnan(cote_home) or np.isnan(cote_away):
                continue

            # 🔹 Filtrer cotes 1.4–1.6
            if 1.4 <= min(cote_home, cote_away) <= 1.6:
                matches.append({
                    "Match": f"{home} vs {away}",
                    "home_team": home,
                    "away_team": away,
                    "cote_home": cote_home,
                    "cote_away": cote_away,
                    "Date": match_time,
                    "Championnat": sport_key
                })
        except:
            continue

    return pd.DataFrame(matches)

# ---------------------------
# 🔹 Interface Streamlit
# ---------------------------
st.set_page_config(page_title="Matchs Safe API-Odds", layout="wide")
st.title("⚽ Analyse Matchs Safe du Week-end (API-Odds)")
st.caption("Basée sur cotes réelles des bookmakers (1X2)")

if st.button("Lancer l'analyse 🧠"):
    all_upcoming = pd.DataFrame()
    for nom, sport_key in CHAMPIONNATS.items():
        up = get_upcoming_matches(sport_key)
        up["Championnat"] = nom
        all_upcoming = pd.concat([all_upcoming, up])
    
    if all_upcoming.empty:
        st.warning("Aucun match safe trouvé pour le week-end avec cotes 1.4–1.6.")
    else:
        # 🔹 Calcul score sécurité et vainqueur probable
        all_upcoming["Winner"] = np.where(all_upcoming["cote_home"] < all_upcoming["cote_away"],
                                          all_upcoming["home_team"], all_upcoming["away_team"])
        all_upcoming["Score_Sécurité"] = (1 - abs(all_upcoming["cote_home"]-all_upcoming["cote_away"]))*100

        top = all_upcoming.sort_values(by="Score_Sécurité", ascending=False).head(4)
        st.success("🏆 Les 3–4 matchs les plus sûrs du week-end :")
        st.dataframe(top[["Championnat","Match","Winner","Score_Sécurité","Date"]], use_container_width=True)

        st.download_button(
            "📥 Télécharger tous les résultats (CSV)",
            all_upcoming.to_csv(index=False).encode("utf-8"),
            "matchs_safe_odds.csv",
            "text/csv"
        )
