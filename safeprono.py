import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
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

SEASON = 2024
HEADERS = {"x-apisports-key": API_KEY}

# ---------------------------
# 🔹 Récupérer les matchs du week-end
# ---------------------------
def get_upcoming_matches(league_id):
    url = f"{BASE_URL}fixtures?league={league_id}&season={SEASON}&next=50"
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
            
            # 🔹 Générer cotes simulées réalistes autour de 1.5
            diff_strength = np.random.uniform(-0.5,0.5)
            cote_home = round(1.5 - 0.05*diff_strength,2)
            cote_away = round(1.5 + 0.05*diff_strength,2)
            
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
# 🔹 ML simple basé sur cotes simulées
# ---------------------------
def train_model(df):
    df["diff_cote"] = df["cote_away"] - df["cote_home"]
    X = df[["cote_home","cote_away","diff_cote"]]
    y = (df["cote_home"] < df["cote_away"]).astype(int)  # home gagne si cote_home < cote_away
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# ---------------------------
# 🔹 Interface Streamlit
# ---------------------------
st.set_page_config(page_title="Matchs Safe Gratuit", layout="wide")
st.title("⚽ Matchs Safe du Week-end (Plan Gratuit)")
st.caption("Basé sur API-Football + cotes simulées réalistes")

if st.button("Lancer l'analyse 🧠"):
    all_upcoming = pd.DataFrame()
    for nom, league_id in CHAMPIONNATS.items():
        up = get_upcoming_matches(league_id)
        up["Championnat"] = nom
        all_upcoming = pd.concat([all_upcoming, up])
    
    if all_upcoming.empty:
        st.warning("Aucun match trouvé pour le week-end.")
    else:
        model, scaler = train_model(all_upcoming)
        X_pred = scaler.transform(all_upcoming[["cote_home","cote_away","diff_cote"]])
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
