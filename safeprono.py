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
# ⚙️ Configuration API
# ---------------------------
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://v3.football.api-sports.io/"

CHAMPIONNATS = {
    "🇫🇷 Ligue 1": 61,
    "🏴 Premier League": 39,
    "🇪🇸 La Liga": 140
}

SEASON = 2024  # Saison actuelle
HEADERS = {"x-apisports-key": API_KEY}

# ---------------------------
# 🔹 Récupérer les matchs passés (historiques) pour entraînement ML
# ---------------------------
def get_historical_data(league_id, season=SEASON, limit=200):
    url = f"{BASE_URL}fixtures?league={league_id}&season={season}&status=FT"
    res = requests.get(url, headers=HEADERS)
    
    st.write(f"📡 Requête historique pour ligue {league_id}... Statut HTTP: {res.status_code}")
    if res.status_code != 200:
        st.error(f"Erreur API : {res.status_code}")
        return pd.DataFrame()
    
    data = res.json().get("response", [])
    matches = []
    
    for m in data[:limit]:
        try:
            home = m["teams"]["home"]["name"]
            away = m["teams"]["away"]["name"]
            winner_home = 1 if m["teams"]["home"]["winner"] else 0
            goals_home = m["goals"]["home"]
            goals_away = m["goals"]["away"]

            # 🔹 Générer des cotes réalistes pour ML
            diff = goals_home - goals_away
            cote_home = round(max(1.3, min(2.8, 1.5 - 0.1*diff)),2)
            cote_away = round(max(1.3, min(2.8, 1.5 + 0.1*diff)),2)
            
            matches.append({
                "home_team": home,
                "away_team": away,
                "home_goals": goals_home,
                "away_goals": goals_away,
                "winner_home": winner_home,
                "cote_home": cote_home,
                "cote_away": cote_away
            })
        except:
            continue
    
    df = pd.DataFrame(matches)
    st.write(f"✅ {len(df)} matchs historiques chargés pour la ligue {league_id}.")
    return df

# ---------------------------
# 🔹 Entraîner modèle ML
# ---------------------------
def train_model(df):
    df["diff_cote"] = df["cote_away"] - df["cote_home"]
    X = df[["cote_home","cote_away","diff_cote"]]
    y = df["winner_home"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, scaler, acc

# ---------------------------
# 🔹 Récupérer les matchs du week-end à venir
# ---------------------------
def get_upcoming_matches(league_id):
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
            
            # 🔹 Générer des cotes réalistes pour plan gratuit
            diff_strength = np.random.uniform(-1,1)
            cote_home = round(max(1.3, min(2.8, 1.5 - 0.1*diff_strength)),2)
            cote_away = round(max(1.3, min(2.8, 1.5 + 0.1*diff_strength)),2)
            
            if 1.4 <= min(cote_home,cote_away) <= 1.6:
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
st.set_page_config(page_title="Analyse Matchs Safe Free", layout="wide")
st.title("⚽ Analyse Matchs Safe du week-end (Plan Gratuit)")
st.caption("Basée sur API-Football + ML + cotes simulées réalistes")

if st.button("Lancer l'analyse 🧠"):
    all_hist = pd.DataFrame()
    for nom, league_id in CHAMPIONNATS.items():
        hist = get_historical_data(league_id)
        all_hist = pd.concat([all_hist, hist])
    
    if all_hist.empty:
        st.error("⚠️ Aucune donnée historique chargée. Vérifie ta clé API et tes IDs ligues.")
    else:
        model, scaler, acc = train_model(all_hist)
        st.info(f"Modèle entraîné avec précision : {round(acc*100,1)}%")
        
        all_upcoming = pd.DataFrame()
        for nom, league_id in CHAMPIONNATS.items():
            up = get_upcoming_matches(league_id)
            up["Championnat"] = nom
            all_upcoming = pd.concat([all_upcoming, up])
        
        if all_upcoming.empty:
            st.warning("Aucun match safe trouvé pour le week-end.")
        else:
            X_pred = scaler.transform(all_upcoming[["cote_home","cote_away","cote_away"]])
            probs = model.predict_proba(X_pred)[:,1]
            all_upcoming["Score_Sécurité"] = (1 - abs(all_upcoming["cote_home"]-all_upcoming["cote_away"]))*probs*100
            all_upcoming["Winner"] = np.where(all_upcoming["cote_home"] < all_upcoming["cote_away"],
                                              all_upcoming["home_team"], all_upcoming["away_team"])
            
            top = all_upcoming.sort_values(by="Score_Sécurité", ascending=False).head(4)
            st.success("🏆 Les 3–4 matchs les plus sûrs du week-end :")
            st.dataframe(top[["Championnat","Match","Winner","Score_Sécurité","Date"]], use_container_width=True)
            
            st.download_button(
                "📥 Télécharger tous les résultats (CSV)",
                all_upcoming.to_csv(index=False).encode("utf-8"),
                "matchs_safe.csv",
                "text/csv"
            )
