from pathlib import Path
 
import pandas as pd
import streamlit as st
 
from model import (
    HISTORY_FEATURES,
    HISTORY_LENGTH,
    OUTCOMES,
    count_matches_with_sufficient_history,
    generate_random_history,
    history_from_match,
    make_match_features,
    matches_with_sufficient_history,
    predict_probabilities,
    summarize_history,
    train_model,
)
 
 
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAINING_DATA = DATA_DIR / "train.csv"
SCORE_DATA = DATA_DIR / "train_target_and_scores.csv"
CHRONOLOGICAL_TRAIN_SHARE = 0.70
 
st.set_page_config(
    page_title="Prédiction de matchs de football",
    page_icon="⚽",
    layout="wide",
)
 
 
@st.cache_resource(show_spinner=False)
def load_model():
    return train_model(TRAINING_DATA)
 
 
@st.cache_resource(show_spinner=False)
def load_historical_model(cutoff_date: str):
    return train_model(TRAINING_DATA, before_date=cutoff_date)
 
 
@st.cache_data(show_spinner=False)
def load_historical_matches() -> tuple[pd.DataFrame, pd.Timestamp]:
    columns = [
        "id",
        "target",
        "home_team_name",
        "away_team_name",
        "match_date",
        "league_name",
        "league_id",
        *HISTORY_FEATURES,
    ]
    matches = pd.read_csv(TRAINING_DATA, usecols=columns)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    if matches["match_date"].isna().any():
        raise ValueError("Certaines dates de match sont invalides.")
 
    ordered_dates = matches["match_date"].sort_values(ignore_index=True)
    cutoff_index = int(len(ordered_dates) * CHRONOLOGICAL_TRAIN_SHARE)
    cutoff_date = ordered_dates.iloc[cutoff_index]
 
    score_data = pd.read_csv(SCORE_DATA, usecols=["id", "score"])
    matches = matches.merge(score_data, on="id", how="left", validate="one_to_one")
    if matches["score"].isna().any():
        raise ValueError("Le score réel manque pour certains matchs historiques.")
 
    eligible = matches_with_sufficient_history(matches)
    validation_matches = matches.loc[
        eligible & matches["match_date"].ge(cutoff_date)
    ].copy()
    validation_matches["league_name"] = validation_matches["league_name"].fillna(
        "Compétition non renseignée"
    )
    return validation_matches, cutoff_date
 
 
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
 
 
def render_historical_team_history(
    team_label: str, history: list[tuple[int, int] | None]
) -> None:
    st.subheader(team_label)
    rows = []
    for index, score in enumerate(history, start=1):
        if score is None:
            rows.append({"Match": f"n-{index}", "Score": "—", "Résultat": "Inconnu"})
            continue
        goals, conceded = score
        if goals > conceded:
            result = "Victoire"
        elif goals == conceded:
            result = "Nul"
        else:
            result = "Défaite"
        rows.append(
            {
                "Match": f"n-{index}",
                "Score": f"{goals} – {conceded}",
                "Résultat": result,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    render_team_statistics(history)
 
 
def outcome_labels(home_team: str, away_team: str) -> dict[str, str]:
    return {
        "home": f"Victoire {home_team}",
        "draw": "Match nul",
        "away": f"Victoire {away_team}",
    }
 
 
def render_prediction_result(
    probabilities: pd.Series, home_team: str, away_team: str
) -> str:
    labels = outcome_labels(home_team, away_team)
    most_likely = str(probabilities.idxmax())
    st.subheader(f"{home_team} (domicile) — {away_team} (extérieur)")
    result_columns = st.columns(3)
    for column, outcome in zip(result_columns, OUTCOMES):
        column.metric(labels[outcome], f"{probabilities[outcome]:.1%}")
        column.progress(float(probabilities[outcome]))
    st.success(
        f"Scénario le plus probable : **{labels[most_likely]}** "
        f"({probabilities[most_likely]:.1%})."
    )
    return most_likely
 
 
def manual_prediction_mode(classifier) -> None:
 
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
 
 
def historical_replay_mode() -> None:
    if not SCORE_DATA.is_file():
        st.error(f"Fichier des scores réels introuvable : {SCORE_DATA}")
        return
 
    with st.spinner("Préparation des matchs historiques…"):
        try:
            matches, cutoff_date = load_historical_matches()
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Impossible de charger les matchs historiques : {error}")
            return
 
    st.info(
        "Ce mode utilise une vraie séparation chronologique : le modèle apprend "
        f"uniquement sur les matchs antérieurs au **{cutoff_date:%d/%m/%Y à %H:%M}**. "
        "Les matchs proposés appartiennent à la période suivante et leur résultat "
        "reste caché jusqu'à sa révélation."
    )
 
    league_counts = matches.groupby(["league_id", "league_name"]).size().sort_values(
        ascending=False
    )
    league_options = list(league_counts.index)
    selected_league = st.selectbox(
        "Compétition",
        options=league_options,
        format_func=lambda league: (
            f"{league[1]} — {int(league_counts.loc[league])} matchs"
        ),
    )
    league_matches = matches.loc[
        matches["league_id"].eq(selected_league[0])
        & matches["league_name"].eq(selected_league[1])
    ]
 
    teams = sorted(
        set(league_matches["home_team_name"].dropna())
        | set(league_matches["away_team_name"].dropna())
    )
    all_teams_label = "Toutes les équipes"
    selected_team = st.selectbox("Équipe", options=[all_teams_label, *teams])
    if selected_team == all_teams_label:
        candidate_matches = league_matches
    else:
        candidate_matches = league_matches.loc[
            league_matches["home_team_name"].eq(selected_team)
            | league_matches["away_team_name"].eq(selected_team)
        ]
    candidate_matches = candidate_matches.sort_values("match_date", ascending=False)
 
    match_labels = {
        int(row.id): (
            f"{row.match_date:%d/%m/%Y} — {row.home_team_name} vs {row.away_team_name}"
        )
        for row in candidate_matches.itertuples()
    }
    selected_match_id = st.selectbox(
        "Match à rejouer",
        options=list(match_labels),
        format_func=match_labels.__getitem__,
    )
    selected_match = candidate_matches.loc[
        candidate_matches["id"].eq(selected_match_id)
    ].iloc[0]
    home_team = str(selected_match["home_team_name"])
    away_team = str(selected_match["away_team_name"])
    home_history = history_from_match(selected_match, "home")
    away_history = history_from_match(selected_match, "away")
 
    st.caption(
        f"Match #{selected_match_id} · {selected_match['league_name']} · "
        f"{selected_match['match_date']:%d/%m/%Y à %H:%M}"
    )
    history_columns = st.columns(2)
    with history_columns[0]:
        render_historical_team_history(f"{home_team} — domicile", home_history)
    with history_columns[1]:
        render_historical_team_history(f"{away_team} — extérieur", away_history)
 
    predict_clicked = st.button(
        "Prédire ce match", type="primary", use_container_width=True
    )
    if predict_clicked:
        try:
            with st.spinner("Entraînement sur la période passée et calcul…"):
                historical_model = load_historical_model(cutoff_date.isoformat())
                features = make_match_features(home_history, away_history)
                probabilities = predict_probabilities(
                    historical_model, features
                ).iloc[0]
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["historical_prediction"] = {
                "match_id": int(selected_match_id),
                "probabilities": probabilities.to_dict(),
                "revealed": False,
            }
 
    prediction_state = st.session_state.get("historical_prediction")
    if prediction_state and prediction_state["match_id"] == int(selected_match_id):
        probabilities = pd.Series(prediction_state["probabilities"])
        predicted_outcome = render_prediction_result(
            probabilities, home_team, away_team
        )
 
        if not prediction_state["revealed"]:
            if st.button("Révéler le résultat réel", use_container_width=True):
                prediction_state["revealed"] = True
                st.session_state["historical_prediction"] = prediction_state
                session_results = st.session_state.setdefault(
                    "historical_session_results", {}
                )
                session_results[int(selected_match_id)] = (
                    predicted_outcome == selected_match["target"]
                )
 
        if prediction_state["revealed"]:
            labels = outcome_labels(home_team, away_team)
            actual_outcome = str(selected_match["target"])
            prediction_is_correct = predicted_outcome == actual_outcome
            message = (
                f"Résultat réel : **{home_team} {selected_match['score']} {away_team}** "
                f"— {labels[actual_outcome]}."
            )
            if prediction_is_correct:
                st.success(f"✅ Bonne prédiction. {message}")
            else:
                st.warning(
                    f"❌ Prédiction incorrecte. {message} "
                    "Le modèle avait choisi "
                    f"**{outcome_labels(home_team, away_team)[predicted_outcome]}**."
                )
 
    session_results = st.session_state.get("historical_session_results", {})
    if session_results:
        correct_predictions = sum(session_results.values())
        score_columns = st.columns(3)
        score_columns[0].metric("Matchs révélés", len(session_results))
        score_columns[1].metric("Bonnes prédictions", correct_predictions)
        score_columns[2].metric(
            "Précision de la session",
            f"{correct_predictions / len(session_results):.1%}",
        )
 
 
def prediction_page() -> None:
    st.title("⚽ Prédire un match")
    st.write(
        "Estimez les probabilités de victoire à domicile, de match nul et de "
        "victoire à l'extérieur à partir des matchs précédents des équipes."
    )
    mode = st.radio(
        "Mode",
        options=("Saisie libre", "Rejouer un match historique"),
        horizontal=True,
    )
 
    if mode == "Rejouer un match historique":
        historical_replay_mode()
        return
 
    st.info(
        "La méthode reprend la partie 4 du notebook : une régression logistique "
        "utilise les points cumulés sur les matchs 1–5 et 6–10 (3 points par "
        "victoire, 1 par nul, 0 par défaite). Au moins 3 scores connus sont "
        "nécessaires dans chacune de ces fenêtres, pour chaque équipe."
    )
    with st.spinner("Entraînement du modèle sur les matchs disponibles…"):
        classifier = load_model()
    st.caption(
        "Le mode libre entraîne le modèle sur les matchs admissibles de "
        "`data/train.csv`. Utilisez le mode historique pour une évaluation "
        "avec séparation chronologique."
    )
    manual_prediction_mode(classifier)
 
 
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