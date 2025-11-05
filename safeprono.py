import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# ⚙️ CONFIG
# ---------------------------
API_KEY = "8b95c22ea5fe5a230b130b83e21a7535"  # ta clé API-Football
BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {"x-apisports-key": API_KEY}

CHAMPIONNATS = {
    "🇫🇷 Ligue 1": 61,     # IDs API-Football
    "🏴 Premier League": 39,
    "🇪🇸 La Liga": 140
}

# ---------------------------
# 📡 Récupération de matchs passés
# ---------------------------
def get_historical_data(league_id, season=2024, limit=200):
    url = f"{BASE_URL}fixtures?league={league_id}&season={season}&status=FT"
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        return pd.DataFrame()

    data = res.json().get("response", [])
    matches = []
    for m in data[:limit]:
        try:
            home = m["teams"]["home"]["name"]
            away = m["teams"]["away"]["name"]
            win = 1 if m["teams"]["home"]["winner"] else 0
            goals_home = m["goals"]["home"]
            goals_away = m["goals"]["away"]
            cote_home = np.random.uniform(1.3, 2.8)  # si tu veux, tu peux utiliser endpoint odds
            cote_away = np.random.uniform(1.3, 2.8)
            matches.append({
                "home_team": home,
                "away_team": away,
                "home_goals": goals_home,
                "away_goals": goals_away,
                "winner_home": win,
                "cote_home": cote_home,
                "cote_away": cote_away
            })
        except:
            continue
    return pd.DataFrame(matches)

# ---------------------------
# 🧠 Entraînement du modèle
# ---------------------------
def train_model(df):
    df["diff_cote"] = df["cote_away"] - df["cote_home"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    X = df[["cote_home", "cote_away", "diff_cote"]]
    y = df["winner_home"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, scaler, acc

# ---------------------------
# 📅 Récupération des matchs du week-end
# ---------------------------
def get_upcoming_matches(league_id):
    next_week = datetime.now() + timedelta(days=7)
    url = f"{BASE_URL}fixtures?league={league_id}&season=2024&next=20"
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        return pd.DataFrame()

    data = res.json().get("response", [])
    matches = []
    for m in data:
        try:
            home = m["teams"]["home"]["name"]
            away = m["teams"]["away"]["name"]
            date = m["fixture"]["date"]
            cote_home = np.random.uniform(1.3, 2.8)  # en prod tu peux remplacer par endpoint odds
            cote_away = np.random.uniform(1.3, 2.8)
            if 1.4 <= min(cote_home, cote_away) <= 1.6:
                matches.append({
                    "Match": f"{home} vs {away}",
                    "home_team": home,
                    "away_team": away,
                    "cote_home": cote_home,
                    "cote_away": cote_away,
                    "Date": date
                })
        except:
            continue
    return pd.DataFrame(matches)

# ---------------------------
# 🖥️ Interface Streamlit
# ---------------------------
st.set_page_config(page_title="Analyse IA des matchs safe", layout="wide")
st.title("⚽ Analyse IA des matchs safe du week-end")
st.caption("Analyse basée sur des données réelles via API-Football + Machine Learning")

if st.button("Lancer l’analyse 🧠"):
    all_hist = pd.DataFrame()
    for nom, id_ in CHAMPIONNATS.items():
        st.write(f"📥 Chargement de l'historique {nom}...")
        hist = get_historical_data(id_)
        all_hist = pd.concat([all_hist, hist])

    if all_hist.empty:
        st.error("Aucune donnée historique chargée.")
    else:
        st.success(f"{len(all_hist)} matchs historiques récupérés.")
        model, scaler, acc = train_model(all_hist)
        st.info(f"Modèle entraîné avec précision : **{round(acc*100,1)}%**")

        all_upcoming = pd.DataFrame()
        for nom, id_ in CHAMPIONNATS.items():
            up = get_upcoming_matches(id_)
            up["Championnat"] = nom
            all_upcoming = pd.concat([all_upcoming, up])

        if all_upcoming.empty:
            st.warning("Aucun match trouvé pour ce week-end.")
        else:
            X_pred = scaler.transform(all_upcoming[["cote_home", "cote_away", "cote_away"]])
            probs = model.predict_proba(X_pred)[:,1]
            all_upcoming["Score_Sécurité"] = (1 - abs(all_upcoming["cote_home"] - all_upcoming["cote_away"])) * probs * 100
            all_upcoming["Winner"] = np.where(all_upcoming["cote_home"] < all_upcoming["cote_away"], all_upcoming["home_team"], all_upcoming["away_team"])
            all_upcoming = all_upcoming.sort_values(by="Score_Sécurité", ascending=False)

            top = all_upcoming.head(4)
            st.success("🏆 Les 4 matchs les plus sûrs du week-end :")
            st.dataframe(top[["Championnat", "Match", "Winner", "Score_Sécurité", "Date"]], use_container_width=True)
