import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analyse Matchs Pro", layout="wide")
st.title("⚽ Analyse Avancée des Matchs")
st.caption("Entrez vos matchs, cotes et statistiques pour obtenir les matchs les plus sûrs")

# ---------------------------
# Fonction pour calculer score sécurité
# ---------------------------
def calculate_score(df):
    """
    Score combinant :
    - Différence de cotes (favoritisme)
    - Forme récente pondérée
    - Différence de buts (optionnelle)
    """
    # Différence de cotes (plus petite = plus sûr)
    df["diff_cote"] = abs(df["cote_home"] - df["cote_away"])
    
    # Forme récente
    df["home_form"] = df["home_wins"]*3 + df["home_draws"]*1 - df["home_losses"]*1
    df["away_form"] = df["away_wins"]*3 + df["away_draws"]*1 - df["away_losses"]*1
    
    # Différence de buts (facultatif)
    df["goal_diff"] = (df["home_goals_scored"] - df["home_goals_against"]) - (df["away_goals_scored"] - df["away_goals_against"])
    
    # Score de sécurité combiné
    # pondération : 50% cotes, 30% forme, 20% goal_diff
    df["score_securite"] = (1 - df["diff_cote"]/10)*50 + ((df["home_form"] - df["away_form"])/20)*30 + ((df["goal_diff"]+10)/20)*20
    
    # Déterminer vainqueur probable
    df["Winner"] = np.where(df["cote_home"] < df["cote_away"], df["home_team"], df["away_team"])
    
    return df

# ---------------------------
# Saisie manuelle via formulaire
# ---------------------------
st.header("Ajouter un match")
with st.form("match_form", clear_on_submit=True):
    home_team = st.text_input("Équipe Domicile")
    away_team = st.text_input("Équipe Extérieur")
    cote_home = st.number_input("Cote Domicile", min_value=1.01, max_value=10.0, value=1.5, step=0.01)
    cote_away = st.number_input("Cote Extérieur", min_value=1.01, max_value=10.0, value=1.5, step=0.01)
    
    st.subheader("Historique de l'équipe Domicile")
    home_wins = st.number_input("Victoires", min_value=0, max_value=50, value=0)
    home_draws = st.number_input("Nuls", min_value=0, max_value=50, value=0)
    home_losses = st.number_input("Défaites", min_value=0, max_value=50, value=0)
    home_goals_scored = st.number_input("Buts marqués", min_value=0, max_value=200, value=0)
    home_goals_against = st.number_input("Buts encaissés", min_value=0, max_value=200, value=0)
    
    st.subheader("Historique de l'équipe Extérieur")
    away_wins = st.number_input("Victoires", min_value=0, max_value=50, value=0)
    away_draws = st.number_input("Nuls", min_value=0, max_value=50, value=0)
    away_losses = st.number_input("Défaites", min_value=0, max_value=50, value=0)
    away_goals_scored = st.number_input("Buts marqués", min_value=0, max_value=200, value=0)
    away_goals_against = st.number_input("Buts encaissés", min_value=0, max_value=200, value=0)
    
    submitted = st.form_submit_button("Ajouter le match")
    
    if submitted:
        if "matches" not in st.session_state:
            st.session_state.matches = []
        st.session_state.matches.append({
            "home_team": home_team,
            "away_team": away_team,
            "cote_home": cote_home,
            "cote_away": cote_away,
            "home_wins": home_wins,
            "home_draws": home_draws,
            "home_losses": home_losses,
            "home_goals_scored": home_goals_scored,
            "home_goals_against": home_goals_against,
            "away_wins": away_wins,
            "away_draws": away_draws,
            "away_losses": away_losses,
            "away_goals_scored": away_goals_scored,
            "away_goals_against": away_goals_against
        })
        st.success(f"Match {home_team} vs {away_team} ajouté !")

# ---------------------------
# Analyse et affichage
# ---------------------------
if "matches" in st.session_state and len(st.session_state.matches) > 0:
    df = pd.DataFrame(st.session_state.matches)
    df = calculate_score(df)
    
    st.header("Analyse des Matchs")
    st.dataframe(df[["home_team","away_team","cote_home","cote_away","Winner","score_securite"]].sort_values(by="score_securite", ascending=False))
    
    st.subheader("🏆 Top 3–4 Matchs les plus sûrs")
    top = df.sort_values(by="score_securite", ascending=False).head(4)
    st.dataframe(top[["home_team","away_team","Winner","score_securite"]])
    
    st.download_button(
        "📥 Télécharger les résultats en CSV",
        df.to_csv(index=False).encode("utf-8"),
        "matchs_analyse_pro.csv",
        "text/csv"
    )
