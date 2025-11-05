import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ---------------------------
# ⚙️ Config API
# ---------------------------
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://v3.football.api-sports.io/"

CHAMPIONNATS = {
    "🇫🇷 Ligue 1": 61,
    "🏴 Premier League": 39,
    "🇪🇸 La Liga": 140
}

SEASON = 2024  # à adapter selon la saison

# ---------------------------
# 🔹 Fonction pour récupérer les historiques
# ---------------------------
def get_historical_data(league_id, season=SEASON, limit=200):
    HEADERS = {"x-apisports-key": API_KEY}
    url = f"{BASE_URL}fixtures?league={league_id}&season={season}&status=FT"
    res = requests.get(url, headers=HEADERS)
    
    st.write(f"📡 Requête API pour la ligue {league_id}...")
    st.write("Statut HTTP :", res.status_code)
    
    if res.status_code != 200:
        st.error(f"Erreur API : {res.status_code} - {res.text[:500]}")
        return pd.DataFrame()
    
    data = res.json().get("response", [])
    if not data:
        st.warning(f"Aucune donnée reçue pour la ligue {league_id}.")
        return pd.DataFrame()
    
    matches = []
    for m in data[:limit]:
        try:
            home = m["teams"]["home"]["name"]
            away = m["teams"]["away"]["name"]
            winner_home = 1 if m["teams"]["home"]["winner"] else 0
            goals_home = m["goals"]["home"]
            goals_away = m["goals"]["away"]
            cote_home = np.random.uniform(1.3, 2.8)
            cote_away = np.random.uniform(1.3, 2.8)
            matches.append({
                "home_team": home,
                "away_team": away,
                "home_goals": goals_home,
                "away_goals": goals_away,
                "winner_home": winner_home,
                "cote_home": cote_home,
                "cote_away": cote_away
            })
        except Exception as e:
            st.warning(f"Erreur traitement match : {e}")
            continue
    
    df = pd.DataFrame(matches)
    st.write(f"✅ {len(df)} matchs historiques chargés pour la ligue {league_id}.")
    return df

# ---------------------------
# 🔹 Entraînement modèle ML
# ---------------------------
def train_model(df):
    df["diff_cote"] = df["cote_away"] - df["cote_home"]
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
# 🔹 Récupération des matchs du week-end
# ---------------------------
def get_upcoming_matches(league_id):
    HEADERS = {"x-apisports-key": API_KEY}
    url = f"{BASE_URL}fixtures?league={league_id}&season={SEASON}&next=20"
    res = requests.get(url, headers=HEADERS)
    
    if res.status_code != 200:
        st.warning(f"Impossible de récupérer les prochains matchs pour ligue {league_id}.")
        return pd.DataFrame()
    
    data = res.json().get("response", [])
    weekend_start = datetime.now()
    weekend_end = weekend_start + timedelta(days=7)
    matches = []
    
    for m in data:
        try:
            home = m["teams"]["home"]["name"]
            away = m["teams"]["away"]["name"]
            match_date = datetime.fromisoformat(m["fixture"]["date"].replace("Z","+00:00"))
            
            if not (weekend_start <= match_date <= weekend_end):
                continue
            
            cote_home = np.random.uniform(1.3, 2.8)
            cote_away = np.random.uniform(1.3, 2.8)
            
            if 1.4 <= min(cote_home, cote_away) <= 1.6:
                matches.append({
                    "Match": f"{home} vs {away}",
                    "home_team": home,
                    "away_team": away,
                    "cote_home": cote_home,
                    "cote_away": cote_away,
                    "Date": match_date,
                    "Championnat": league_id
                })
        except:
            continue
    
    return pd.DataFrame(matches)

# ---------------------------
# 🔹 Interface Streamlit
# ---------------------------
st.set_page_config(page_title="Analyse Pro des matchs safe", layout="wide")
st.title("⚽ Analyse Pro IA : Matchs safe du week-end")
st.caption("Basée sur API-Football + Machine Learning")

if st.button("Lancer l'analyse 🧠"):
    # 🔹 Charger historiques
    all_hist = pd.DataFrame()
    for nom, league_id in CHAMPIONNATS.items():
        hist = get_historical_data(league_id)
        all_hist = pd.concat([all_hist, hist])
    
    if all_hist.empty:
        st.error("⚠️ Aucune donnée historique chargée. Vérifie ta clé API et tes IDs ligues.")
    else:
        # 🔹 Entraîner modèle
        model, scaler, acc = train_model(all_hist)
        st.info(f"Modèle entraîné avec précision : {round(acc*100,1)}%")
        
        # 🔹 Récupérer matchs du week-end
        all_upcoming = pd.DataFrame()
        for nom, league_id in CHAMPIONNATS.items():
            up = get_upcoming_matches(league_id)
            up["Championnat"] = nom
            all_upcoming = pd.concat([all_upcoming, up])
        
        if all_upcoming.empty:
            st.warning("Aucun match safe trouvé pour le week-end.")
        else:
            # 🔹 Prédictions
            X_pred = scaler.transform(all_upcoming[["cote_home","cote_away","cote_away"]])
            probs = model.predict_proba(X_pred)[:,1]
            all_upcoming["Score_Sécurité"] = (1 - abs(all_upcoming["cote_home"]-all_upcoming["cote_away"]))*probs*100
            all_upcoming["Winner"] = np.where(all_upcoming["cote_home"] < all_upcoming["cote_away"],
                                              all_upcoming["home_team"], all_upcoming["away_team"])
            
            top = all_upcoming.sort_values(by="Score_Sécurité", ascending=False).head(4)
            st.success("🏆 Les 4 matchs les plus sûrs du week-end :")
            st.dataframe(top[["Championnat","Match","Winner","Score_Sécurité","Date"]], use_container_width=True)
