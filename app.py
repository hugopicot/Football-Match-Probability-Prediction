from pathlib import Path

import pandas as pd
import streamlit as st

from model import (
    HISTORY_FEATURES,
    HISTORY_LENGTH,
    OUTCOMES,
    count_matches_with_sufficient_history,
    generate_random_history,
    make_match_features,
    predict_probabilities,
    summarize_history,
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


def make_history_table(
    scores: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    scores_are_known = scores is not None
    resolved_scores = scores or [(0, 0)] * HISTORY_LENGTH
    return pd.DataFrame(
        {
            "Match": [f"Match {number}" for number in range(1, HISTORY_LENGTH + 1)],
            "Match renseigné": [scores_are_known] * HISTORY_LENGTH,
            "Buts marqués": [score[0] for score in resolved_scores],
            "Buts encaissés": [score[1] for score in resolved_scores],
        },
    )


def randomize_histories() -> None:
    for team_key in ("team_a", "team_b"):
        st.session_state[f"{team_key}_history_data"] = make_history_table(
            generate_random_history()
        )
    st.session_state["history_editor_version"] = (
        st.session_state.get("history_editor_version", 0) + 1
    )


def render_team_statistics(history: list[tuple[int, int] | None]) -> None:
    summary = summarize_history(history)
    st.markdown("##### Données calculées")
    if not summary.matches:
        st.caption("Aucun score renseigné pour le moment.")
        return

    metric_columns = st.columns(2)
    metric_columns[0].metric(
        "Bilan",
        f"{summary.wins} V · {summary.draws} N · {summary.losses} D",
    )
    metric_columns[1].metric(
        "Buts",
        f"{summary.goals_for} marqués",
        delta=f"{summary.goals_against} encaissés",
        delta_color="off",
    )
    metric_columns[0].metric(
        "Points",
        f"{summary.points} / {summary.matches * 3}",
        delta=f"{summary.goal_difference:+d} de différence de buts",
    )
    points_per_match = summary.recent_points_per_match
    metric_columns[1].metric(
        "Forme récente",
        summary.form_level,
        delta=(
            f"{points_per_match:.1f} pt/match" if points_per_match is not None else None
        ),
        delta_color="off",
    )


def render_team_history(team_label: str, team_key: str) -> list[tuple[int, int] | None]:
    st.caption(
        f"Renseignez les 10 derniers matchs de {team_label}. "
        "Cochez uniquement les matchs dont vous connaissez le score."
    )
    history_data_key = f"{team_key}_history_data"
    if history_data_key not in st.session_state:
        st.session_state[history_data_key] = make_history_table()
    editor_version = st.session_state.get("history_editor_version", 0)
    edited_history = st.data_editor(
        st.session_state[history_data_key],
        key=f"{team_key}_history_{editor_version}",
        hide_index=True,
        num_rows="fixed",
        disabled=["Match"],
        use_container_width=True,
        column_config={
            "Match renseigné": st.column_config.CheckboxColumn(
                "Score connu", default=False
            ),
            "Buts marqués": st.column_config.NumberColumn(
                min_value=0, max_value=20, step=1
            ),
            "Buts encaissés": st.column_config.NumberColumn(
                min_value=0, max_value=20, step=1
            ),
        },
    )
    return [
        (int(row["Buts marqués"]), int(row["Buts encaissés"]))
        if row["Match renseigné"]
        else None
        for _, row in edited_history.iterrows()
    ]


def prediction_page() -> None:
    st.title("⚽ Prédire un match")
    st.write(
        "Estimez les probabilités de victoire à domicile, de match nul et de "
        "victoire à l'extérieur à partir des points des matchs précédents des équipes."
    )
    st.info(
        "La méthode reprend la partie 4 du notebook : une régression logistique "
        "utilise les points cumulés sur les matchs 1–5 et 6–10 (3 points par "
        "victoire, 1 par nul, 0 par défaite). Au moins 3 scores connus sont "
        "nécessaires dans chacune de ces fenêtres, pour chaque équipe. "
        "Le notebook mesure 47,2 % de bonnes prédictions sur un test aléatoire "
        "de 30 % ; ce résultat indicatif ne garantit pas la précision d'un match."
    )

    with st.spinner("Entraînement du modèle sur les matchs disponibles…"):
        classifier = load_model()
    st.caption(
        "Le modèle est entraîné sur les matchs de `data/train.csv` qui disposent "
        "d'un historique suffisant."
    )

    st.button(
        "🎲 Générer des scores aléatoires",
        on_click=randomize_histories,
        help="Remplit les 10 matchs des deux équipes avec des scores entre 0 et 5.",
        use_container_width=True,
    )

    with st.container(border=True):
        team_columns = st.columns(2)
        team_a = team_columns[0].text_input("Équipe A", value="Équipe A")
        team_b = team_columns[1].text_input("Équipe B", value="Équipe B")
        team_a_label = team_a or "Équipe A"
        team_b_label = team_b or "Équipe B"
        home_team_key = st.selectbox(
            "Qui joue à domicile ?",
            options=("team_a", "team_b"),
            format_func=lambda team_key: (
                team_a_label if team_key == "team_a" else team_b_label
            ),
            help="L'autre équipe sera automatiquement considérée comme visiteuse.",
        )
        team_a_role = "domicile" if home_team_key == "team_a" else "extérieur"
        team_b_role = "domicile" if home_team_key == "team_b" else "extérieur"
        history_columns = st.columns(2)
        with history_columns[0]:
            st.subheader(f"Forme récente — {team_a_label} ({team_a_role})")
            team_a_history = render_team_history(team_a_label, "team_a")
            render_team_statistics(team_a_history)
        with history_columns[1]:
            st.subheader(f"Forme récente — {team_b_label} ({team_b_role})")
            team_b_history = render_team_history(team_b_label, "team_b")
            render_team_statistics(team_b_history)
        st.caption(
            "Les lignes 1–5 sont les cinq matchs les plus récents ; les lignes "
            "6–10 servent de référence pour la forme habituelle. Le niveau de "
            "forme correspond aux points par match sur les lignes 1–5."
        )
        submitted = st.button(
            "Calculer les probabilités", type="primary", use_container_width=True
        )

    if home_team_key == "team_a":
        home_team, away_team = team_a_label, team_b_label
        home_history, away_history = team_a_history, team_b_history
    else:
        home_team, away_team = team_b_label, team_a_label
        home_history, away_history = team_b_history, team_a_history

    if submitted:
        try:
            match_features = make_match_features(home_history, away_history)
            probabilities = predict_probabilities(classifier, match_features).iloc[0]
        except ValueError as error:
            st.error(str(error))
        else:
            most_likely = probabilities.idxmax()

            st.subheader(f"{home_team} (domicile) — {away_team} (extérieur)")
            result_columns = st.columns(3)
            outcome_labels = {
                "home": f"Victoire {home_team}",
                "draw": "Match nul",
                "away": f"Victoire {away_team}",
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
        "des scores des 10 matchs précédents. Chaque équipe doit avoir au moins "
        "3 scores connus dans les matchs 1–5 et 3 dans les matchs 6–10 pour "
        "utiliser le modèle. Les lignes avec un historique plus incomplet "
        "recevront la répartition moyenne des résultats du jeu d'entraînement."
    )
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            matches = pd.read_csv(
                uploaded_file,
                usecols=lambda column: column == "id" or column in HISTORY_FEATURES,
            )
            eligible_count = count_matches_with_sufficient_history(matches)
            probabilities = predict_probabilities(
                classifier, matches, allow_incomplete_history=True
            )
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Impossible de traiter ce fichier : {error}")
        else:
            fallback_count = len(matches) - eligible_count
            if fallback_count:
                st.warning(
                    f"{fallback_count:,} ligne(s) n'ont pas assez d'historique. "
                    "Pour celles-ci, les probabilités affichées sont la fréquence "
                    "des résultats sur les matchs d'entraînement admissibles."
                )
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
