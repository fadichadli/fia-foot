import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="IA Pronos Football", page_icon="⚽", layout="centered")
st.title("⚽ Application de Pronostics Foot par IA")
st.write("Sélectionnez deux équipes pour obtenir l'analyse, la double chance et le conseil automatique.")

@st.cache_data
def charger_et_calculer_stats():
    df = pd.read_csv("data_ia.csv")
    df['Total_Goals'] = df['FTHG'] + df['FTAG']
    df['Over2.5'] = np.where(df['Total_Goals'] > 2.5, 1, 0)
    
    # Statistiques historiques à DOMICILE
    home_profile = df.groupby('HomeTeam').agg(
        HomeGoalsScored=('FTHG', 'mean'),
        HomeGoalsConceded=('FTAG', 'mean'),
        HomeWinRate=('FTR', lambda x: (x == 'H').mean())
    ).reset_index()
    
    # Statistiques historiques à L'EXTÉRIEUR
    away_profile = df.groupby('AwayTeam').agg(
        AwayGoalsScored=('FTAG', 'mean'),
        AwayGoalsConceded=('FTHG', 'mean'),
        AwayWinRate=('FTR', lambda x: (x == 'A').mean())
    ).reset_index()
    
    df_train = df.merge(home_profile, on='HomeTeam', how='left')
    df_train = df_train.merge(away_profile, on='AwayTeam', how='left')
    df_train = df_train.fillna(0)
    
    features = ['HomeGoalsScored', 'HomeGoalsConceded', 'HomeWinRate', 
                'AwayGoalsScored', 'AwayGoalsConceded', 'AwayWinRate']
    
    return df_train, features, home_profile, away_profile

try:
    df_train, features, home_profile, away_profile = charger_et_calculer_stats()
    X = df_train[features]
    
    model_1X2 = RandomForestClassifier(n_estimators=250, random_state=42, min_samples_leaf=5)
    model_1X2.fit(X, df_train['FTR'])
    
    model_goals = RandomForestClassifier(n_estimators=250, random_state=42, min_samples_leaf=5)
    model_goals.fit(X, df_train['Over2.5'])
    
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
            stats_home = home_profile[home_profile['HomeTeam'] == home_team]
            stats_away = away_profile[away_profile['AwayTeam'] == away_team]
            
            if stats_home.empty or stats_away.empty:
                st.error("Données insuffisantes.")
            else:
                match_features = pd.DataFrame([{
                    'HomeGoalsScored': stats_home['HomeGoalsScored'].values[0],
                    'HomeGoalsConceded': stats_home['HomeGoalsConceded'].values[0],
                    'HomeWinRate': stats_home['HomeWinRate'].values[0],
                    'AwayGoalsScored': stats_away['AwayGoalsScored'].values[0],
                    'AwayGoalsConceded': stats_away['AwayGoalsConceded'].values[0],
                    'AwayWinRate': stats_away['AwayWinRate'].values[0]
                }])
                
                probabilites = model_1X2.predict_proba(match_features)[0]
                proba_dict = dict(zip(model_1X2.classes_, probabilites))
                
                p_H = proba_dict.get('H', 0) * 100
                p_D = proba_dict.get('D', 0) * 100
                p_A = proba_dict.get('A', 0) * 100
                
                # Calcul précis des Doubles Chances
                p_1X = p_H + p_D
                p_X2 = p_D + p_A
                p_12 = p_H + p_A
                
                p_over = model_goals.predict_proba(match_features)[0][1] * 100
                p_under = 100 - p_over

                # --- BLOC DU CONSEIL AUTOMATIQUE DE L'IA ---
                st.markdown("### 💡 Le Conseil Prono de l'IA")
                
                if p_H > 55:
                    st.info(f"🏆 **Pronostic conseillé : Victoire de {home_team}** (Confiance : {p_H:.1f}%)")
                elif p_A > 55:
                    st.info(f"🏆 **Pronostic conseillé : Victoire de {away_team}** (Confiance : {p_A:.1f}%)")
                elif p_1X > 78:
                    st.success(f"🛡️ **Pronostic sécurisé : Double Chance 1X** ({home_team} ou Nul)")
                elif p_X2 > 78:
                    st.success(f"🛡️ **Pronostic sécurisé : Double Chance X2** (Nul ou {away_team})")
                elif p_over > 60:
                    st.warning(f"⚽ **Pronostic Alternatif : Plus de 2.5 buts** (Match très ouvert)")
                else:
                    st.error("⚖️ **Match très indécis : Pas de pronostic fiable recommandé (À éviter)**")
                
                # --- AFFICHAGE DES CHIFFRES EN 3 COLONNES ---
                st.markdown(f"### 📊 Détails des Analyses : {home_team} VS {away_team}")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.markdown("#### 🎯 Résultat (1X2)")
                    st.write(f"🏠 Victoire : **{p_H:.1f}%**")
                    st.write(f"🤝 Nul : **{p_D:.1f}%**")
                    st.write(f"✈️ Victoire : **{p_A:.1f}%**")
                
                with col_res2:
                    st.markdown("#### 🛡️ Double Chance")
                    st.write(f"🤞 **1X** : **{p_1X:.1f}%**")
                    st.write(f"🤞 **X2** : **{p_X2:.1f}%**")
                    st.write(f"💥 **12** : **{p_12:.1f}%**")
                
                with col_res3:
                    st.markdown("#### ⚽ Buts (+/- 2.5)")
                    st.write(f"🔥 Plus de 2.5 : **{p_over:.1f}%**")
                    st.write(f"🥶 Moins de 2.5 : **{p_under:.1f}%**")
                    
except FileNotFoundError:
    st.error("Fichier de données 'data_ia.csv' introuvable.")
