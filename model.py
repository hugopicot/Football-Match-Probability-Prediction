"""Training and prediction helpers for the football match app."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


OUTCOMES = ("home", "draw", "away")
HISTORY_LENGTH = 5
HISTORY_FEATURES = tuple(
    f"{team}_team_history_{feature}_{match_number}"
    for team in ("home", "away")
    for match_number in range(1, HISTORY_LENGTH + 1)
    for feature in ("goal", "opponent_goal", "is_play_home")
)
VENUE_FORM_FEATURES = tuple(
    f"{team}_form_{feature}_{venue}"
    for team, venue in (("home", "home"), ("away", "away"))
    for feature in ("goals", "conceded", "goal_difference", "points", "matches")
)


def _make_venue_form_features(
    features: pd.DataFrame, history_validity: pd.DataFrame
) -> pd.DataFrame:
    """Summarize each team's recent form in the venue of the upcoming match."""
    venue_form: dict[str, pd.Series] = {}
    for team, venue in (("home", "home"), ("away", "away")):
        goals = features.loc[
            :,
            [
                f"{team}_team_history_goal_{number}"
                for number in range(1, HISTORY_LENGTH + 1)
            ],
        ]
        conceded = features.loc[
            :,
            [
                f"{team}_team_history_opponent_goal_{number}"
                for number in range(1, HISTORY_LENGTH + 1)
            ],
        ]
        conceded.columns = goals.columns
        played_at_home = features.loc[
            :,
            [
                f"{team}_team_history_is_play_home_{number}"
                for number in range(1, HISTORY_LENGTH + 1)
            ],
        ].astype(bool)
        matching_venue = played_at_home if venue == "home" else ~played_at_home
        matching_venue.columns = goals.columns
        available_history = history_validity.loc[
            :,
            [f"{team}_history_{number}" for number in range(1, HISTORY_LENGTH + 1)],
        ].copy()
        available_history.columns = goals.columns
        matching_venue &= available_history
        match_count = matching_venue.sum(axis=1)
        denominator = match_count.where(match_count > 0, 1)
        average_goals = goals.where(matching_venue).sum(axis=1) / denominator
        average_conceded = conceded.where(matching_venue).sum(axis=1) / denominator
        points = (
            goals.gt(conceded).astype(float) * 3
            + goals.eq(conceded).astype(float)
        )

        venue_form[f"{team}_form_goals_{venue}"] = average_goals
        venue_form[f"{team}_form_conceded_{venue}"] = average_conceded
        venue_form[f"{team}_form_goal_difference_{venue}"] = (
            average_goals - average_conceded
        )
        venue_form[f"{team}_form_points_{venue}"] = (
            points.where(matching_venue).sum(axis=1) / denominator
        )
        venue_form[f"{team}_form_matches_{venue}"] = match_count

    return pd.DataFrame(venue_form, index=features.index).loc[:, VENUE_FORM_FEATURES]


def prepare_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Select and normalize the recent-match features used by the model."""
    missing = [column for column in HISTORY_FEATURES if column not in matches.columns]
    if missing:
        raise ValueError(
            "Le fichier ne contient pas toutes les colonnes requises. "
            f"Colonnes manquantes : {', '.join(missing)}"
        )

    features = matches.loc[:, HISTORY_FEATURES].copy()
    history_validity = pd.DataFrame(index=matches.index)
    for team in ("home", "away"):
        for number in range(1, HISTORY_LENGTH + 1):
            history_columns = [
                f"{team}_team_history_{feature}_{number}"
                for feature in ("goal", "opponent_goal", "is_play_home")
            ]
            history_validity[f"{team}_history_{number}"] = (
                matches.loc[:, history_columns].notna().all(axis=1)
            )

    for column in HISTORY_FEATURES:
        numeric = pd.to_numeric(features[column], errors="coerce")
        if column.endswith("_is_play_home"):
            boolean_values = (
                features[column]
                .astype("string")
                .str.strip()
                .str.lower()
                .map({"true": 1.0, "false": 0.0})
            )
            numeric = numeric.fillna(boolean_values)
        features[column] = numeric.fillna(0.0)

    return pd.concat(
        [features, _make_venue_form_features(features, history_validity)], axis=1
    )


def train_model(training_data_path: str | Path) -> HistGradientBoostingClassifier:
    """Fit the classifier using the available training data."""
    training_data_path = Path(training_data_path)
    if not training_data_path.is_file():
        raise FileNotFoundError(
            f"Fichier d'entraînement introuvable : {training_data_path}"
        )

    training_data = pd.read_csv(
        training_data_path,
        usecols=[*HISTORY_FEATURES, "target"],
    )
    labels = training_data["target"].astype("string").str.strip().str.lower()
    unexpected = sorted(set(labels.dropna()) - set(OUTCOMES))
    if unexpected:
        raise ValueError(
            "La colonne target contient des valeurs non reconnues : "
            + ", ".join(unexpected)
        )
    if labels.isna().any():
        raise ValueError("La colonne target contient des valeurs manquantes.")

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        random_state=42,
    )
    classifier.fit(prepare_features(training_data), labels)
    return classifier


def make_match_features(
    home_history: list[tuple[int, int, bool]],
    away_history: list[tuple[int, int, bool]],
) -> pd.DataFrame:
    """Build one model row from recent scores and home/away venue flags."""
    row: dict[str, float] = {column: float("nan") for column in HISTORY_FEATURES}
    for team, history in (("home", home_history), ("away", away_history)):
        if len(history) > HISTORY_LENGTH:
            raise ValueError(
                f"Au maximum {HISTORY_LENGTH} matchs récents peuvent être saisis."
            )
        for match_number, (goals, opponent_goals, played_at_home) in enumerate(
            history, start=1
        ):
            if goals < 0 or opponent_goals < 0:
                raise ValueError("Les scores ne peuvent pas être négatifs.")
            row[f"{team}_team_history_goal_{match_number}"] = float(goals)
            row[f"{team}_team_history_opponent_goal_{match_number}"] = float(
                opponent_goals
            )
            row[f"{team}_team_history_is_play_home_{match_number}"] = float(
                played_at_home
            )

    return pd.DataFrame([row], columns=HISTORY_FEATURES)


def predict_probabilities(
    classifier: HistGradientBoostingClassifier,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Return home/draw/away probabilities in a stable column order."""
    predicted = classifier.predict_proba(prepare_features(matches))
    probabilities = pd.DataFrame(0.0, index=matches.index, columns=OUTCOMES)
    for index, outcome in enumerate(classifier.classes_):
        probabilities[outcome] = predicted[:, index]
    return probabilities
