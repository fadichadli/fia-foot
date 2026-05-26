import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="IA Pronos Football", page_icon="⚽", layout="centered")
st.title("⚽ Application de Pronostics Foot par IA")
st.write("Sélectionnez deux équipes pour obtenir les probabilités réelles basées sur leurs statistiques historiques.")

@st.cache_data
def charger_et_calculer_stats():
    df = pd.read_csv("data_ia.csv")
    df['Total_Goals'] = df['FTHG'] + df['FTAG']
    df['Over2.5'] = np.where(df['Total_Goals'] > 2.5, 1, 0)
    
    # 1. Calcul des statistiques historiques à DOMICILE
    home_profile = df.groupby('HomeTeam').agg(
        HomeGoalsScored=('FTHG', 'mean'),
        HomeGoalsConceded=('FTAG', 'mean'),
        HomeWinRate=('FTR', lambda x: (x == 'H').mean())
    ).reset_index()
    
    # 2. Calcul des statistiques historiques à L'EXTÉRIEUR
    away_profile = df.groupby('AwayTeam').agg(
        AwayGoalsScored=('FTAG', 'mean'),
        AwayGoalsConceded=('FTHG', 'mean'),
        AwayWinRate=('FTR', lambda x: (x == 'A').mean())
    ).reset_index()
    
    # 3. Fusion de ces statistiques dans notre tableau principal
    df_train = df.merge(home_profile, on='HomeTeam', how='left')
    df_train = df_train.merge(away_profile, on='AwayTeam', how='left')
    df_train = df_train.fillna(0)
    
    # Nos caractéristiques logiques pour l'IA
    features = ['HomeGoalsScored', 'HomeGoalsConceded', 'HomeWinRate', 
                'AwayGoalsScored', 'AwayGoalsConceded', 'AwayWinRate']
    
    return df_train, features, home_profile, away_profile

try:
    df_train, features, home_profile, away_profile = charger_et_calculer_stats()
    
    X = df_train[features]
    
    # Entraînement de l'IA sur des vraies données de performance
    model_1X2 = RandomForestClassifier(n_estimators=250, random_state=42, min_samples_leaf=5)
    model_1X2.fit(X, df_train['FTR'])
    
    model_goals = RandomForestClassifier(n_estimators=250, random_state=42, min_samples_leaf=5)
    model_goals.fit(X, df_train['Over2.5'])
    
    # Liste unique de toutes les équipes disponibles
    liste_equipes = sorted(list(set(df_train['HomeTeam'].unique()).intersection(set(df_train['AwayTeam'].unique()))))

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Équipe à domicile", liste_equipes)
    with col2:
        away_team = st.selectbox("✈️ Équipe à l'extérieur", liste_equipes)

    if st.button("📊 CALCULER LE PRONOSTIC", use_container_width=True):
        if home_team == away_team:
            st.warning("⚠️ Veuillez choisir deux équipes différentes !")
        else:
            # Récupération du profil des deux équipes sélectionnées
            stats_home = home_profile[home_profile['HomeTeam'] == home_team]
            stats_away = away_profile[away_profile['AwayTeam'] == away_team]
            
            if stats_home.empty or stats_away.empty:
                st.error("Données insuffisantes pour l'une des deux équipes.")
            else:
                # Construction du vecteur de match pour la prédiction
                match_features = pd.DataFrame([{
                    'HomeGoalsScored': stats_home['HomeGoalsScored'].values[0],
                    'HomeGoalsConceded': stats_home['HomeGoalsConceded'].values[0],
                    'HomeWinRate': stats_home['HomeWinRate'].values[0],
                    'AwayGoalsScored': stats_away['AwayGoalsScored'].values[0],
                    'AwayGoalsConceded': stats_away['AwayGoalsConceded'].values[0],
                    'AwayWinRate': stats_away['AwayWinRate'].values[0]
                }])
                
                # Prédiction 1X2
                probabilites = model_1X2.predict_proba(match_features)[0]
                proba_dict = dict(zip(model_1X2.classes_, probabilites))
                
                p_H = proba_dict.get('H', 0) * 100
                p_D = proba_dict.get('D', 0) * 100
                p_A = proba_dict.get('A', 0) * 100
                
                # Doubles Chances
                p_1X = p_H + p_D
                p_X2 = p_D + p_A
                p_12 = p_H + p_A
                
                # Buts
                p_over = model_goals.predict_proba(match_features)[0][1] * 100
                p_under = 100 - p_over

                # Affichage
                st.markdown(f"### 📊 Analyses réalistes : {home_team} VS {away_team}")
                
                st.markdown("#### 🎯 Résultat du Match (1X2)")
                st.write(f"🏠 Victoire {home_team} : **{p_H:.1f}%**")
                st.write(f"🤝 Match Nul : **{p_D:.1f}%**")
                st.write(f"✈️ Victoire {away_team} : **{p_A:.1f}%**")
                
                st.markdown("#### 🛡️ Marchés Double Chance")
                st.success(f"🫵 **1X** ({home_team} ou Nul) : **{p_1X:.1f}%**")
                st.info(f"🫵 **X2** (Nul ou {away_team}) : **{p_X2:.1f}%**")
                st.info(f"🫵 **12** (Pas de match nul) : **{p_12:.1f}%**")
                
                st.markdown("#### ⚽ Nombre de Buts (+/- 2.5)")
                st.warning(f"🔥 Plus de 2.5 buts : **{p_over:.1f}%**")
                st.error(f"🥶 Moins de 2.5 buts : **{p_under:.1f}%**")
                
except FileNotFoundError:
    st.error("Fichier de données 'data_ia.csv' introuvable.")
