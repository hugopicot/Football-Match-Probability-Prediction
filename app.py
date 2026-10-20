from pathlib import Path

import pandas as pd
import streamlit as st

from model import (
    HISTORY_FEATURES,
    HISTORY_LENGTH,
    OUTCOMES,
    make_match_features,
    predict_probabilities,
    train_model,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
TRAINING_DATA = DATA_DIR / "train.csv"

st.set_page_config(
    page_title="Prédiction de matchs de football",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model():
    return train_model(TRAINING_DATA)


def render_team_history(team_label: str, team_key: str) -> list[tuple[int, int, bool]]:
    count = st.select_slider(
        f"Matchs récents renseignés — {team_label}",
        options=list(range(1, HISTORY_LENGTH + 1)),
        value=3,
        key=f"{team_key}_history_count",
    )
    history: list[tuple[int, int, bool]] = []
    for match_number in range(1, count + 1):
        st.markdown(f"**Match récent {match_number}**")
        score_columns = st.columns(3)
        goals = score_columns[0].number_input(
            f"Buts de {team_label}",
            min_value=0,
            max_value=20,
            value=1,
            key=f"{team_key}_goals_{match_number}",
        )
        opponent_goals = score_columns[1].number_input(
            "Buts encaissés",
            min_value=0,
            max_value=20,
            value=1,
            key=f"{team_key}_opponent_goals_{match_number}",
        )
        played_at_home = score_columns[2].checkbox(
            "Jouait à domicile",
            value=True,
            key=f"{team_key}_played_at_home_{match_number}",
        )
        history.append((goals, opponent_goals, played_at_home))

    return history


def prediction_page() -> None:
    st.title("⚽ Prédire un match")
    st.write(
        "Estimez les probabilités de victoire à domicile, de match nul et de "
        "victoire à l'extérieur à partir des cinq derniers matchs de chaque équipe."
    )
    st.info(
        "Le modèle s'entraîne automatiquement sur `data/train.csv` au premier "
        "chargement. Les noms des équipes servent à présenter le résultat ; "
        "la prédiction utilise les scores récents et le lieu des matchs passés. "
        "Elle tient aussi compte de la forme du club recevant à domicile et de "
        "celle du visiteur à l'extérieur."
    )

    with st.spinner("Entraînement du modèle sur les matchs disponibles…"):
        classifier = load_model()

    with st.form("match_prediction"):
        team_columns = st.columns(2)
        home_team = team_columns[0].text_input("Équipe à domicile", value="Équipe A")
        away_team = team_columns[1].text_input("Équipe visiteuse", value="Équipe B")
        history_columns = st.columns(2)
        with history_columns[0]:
            st.subheader(f"Forme récente — {home_team or 'Équipe à domicile'}")
            home_history = render_team_history(home_team or "Équipe à domicile", "home")
        with history_columns[1]:
            st.subheader(f"Forme récente — {away_team or 'Équipe visiteuse'}")
            away_history = render_team_history(away_team or "Équipe visiteuse", "away")
        submitted = st.form_submit_button(
            "Calculer les probabilités", type="primary", use_container_width=True
        )

    if submitted:
        match_features = make_match_features(home_history, away_history)
        probabilities = predict_probabilities(classifier, match_features).iloc[0]
        most_likely = probabilities.idxmax()

        st.subheader(f"{home_team or 'Équipe à domicile'} — {away_team or 'Équipe visiteuse'}")
        result_columns = st.columns(3)
        outcome_labels = {
            "home": f"Victoire {home_team or 'domicile'}",
            "draw": "Match nul",
            "away": f"Victoire {away_team or 'visiteur'}",
        }
        for column, outcome in zip(result_columns, OUTCOMES):
            column.metric(outcome_labels[outcome], f"{probabilities[outcome]:.1%}")
            column.progress(float(probabilities[outcome]))
        st.success(
            f"Scénario le plus probable : **{outcome_labels[most_likely]}** "
            f"({probabilities[most_likely]:.1%})."
        )

    st.divider()
    st.subheader("Prédire un fichier de matchs")
    st.write(
        "Importez un CSV au format des données du projet, avec les colonnes "
        "d'historique récentes requises. Le fichier de résultats peut être téléchargé."
    )
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            matches = pd.read_csv(
                uploaded_file,
                usecols=lambda column: column == "id" or column in HISTORY_FEATURES,
            )
            probabilities = predict_probabilities(classifier, matches)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Impossible de traiter ce fichier : {error}")
        else:
            output = pd.DataFrame(index=matches.index)
            if "id" in matches.columns:
                output["id"] = matches["id"]
            for outcome in OUTCOMES:
                output[outcome] = probabilities[outcome].to_numpy()
            st.dataframe(output.head(100), use_container_width=True)
            st.download_button(
                "Télécharger les prédictions (format sample_submission)",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="predictions_matchs.csv",
                mime="text/csv",
            )


def exploration_page() -> None:
    st.title("📊 Explorer les données")
    if not TRAINING_DATA.is_file():
        st.error(f"Fichier d'entraînement introuvable : {TRAINING_DATA}")
        return

    training_data = pd.read_csv(
        TRAINING_DATA,
        usecols=["target", "league_name", "is_cup"],
    )
    total_matches = len(training_data)
    target_counts = training_data["target"].value_counts().reindex(OUTCOMES, fill_value=0)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Matchs d'entraînement", f"{total_matches:,}".replace(",", " "))
    metric_columns[1].metric("Ligues représentées", f"{training_data['league_name'].nunique():,}".replace(",", " "))
    metric_columns[2].metric(
        "Matchs de coupe",
        f"{int(training_data['is_cup'].fillna(False).astype(bool).sum()):,}".replace(",", " "),
    )

    st.subheader("Répartition des résultats")
    target_summary = pd.DataFrame(
        {
            "Nombre de matchs": target_counts,
            "Part": (target_counts / total_matches * 100).round(1),
        }
    )
    chart_column, table_column = st.columns([2, 1])
    chart_column.bar_chart(target_summary["Nombre de matchs"])
    table_column.dataframe(target_summary, use_container_width=True)

    st.subheader("Ligues les plus représentées")
    league_counts = (
        training_data["league_name"]
        .fillna("Non renseignée")
        .value_counts()
        .head(15)
    )
    st.bar_chart(league_counts)


st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page", ("Prédire un match", "Explorer les données"))
if page == "Prédire un match":
    prediction_page()
else:
    exploration_page()
