"""Training and prediction helpers for the notebook's form-based model."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.linear_model import LogisticRegression


OUTCOMES = ("home", "draw", "away")
HISTORY_LENGTH = 10
MIN_MATCHES_PER_WINDOW = 3
HISTORY_FEATURES = tuple(
    f"{team}_team_history_{feature}_{match_number}"
    for team in ("home", "away")
    for match_number in range(1, HISTORY_LENGTH + 1)
    for feature in ("goal", "opponent_goal")
)
MODEL_FEATURES = ("home_5", "away_5", "home_610", "away_610")
HISTORY_WINDOWS = ((1, 5, "5"), (6, 10, "610"))


@dataclass(frozen=True)
class FormModel:
    classifier: LogisticRegression
    outcome_prior: dict[str, float]


@dataclass(frozen=True)
class TeamHistorySummary:
    matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    points: int
    goal_difference: int
    recent_points_per_match: float | None
    form_level: str


def generate_random_history(
    rng: random.Random | None = None,
    max_goals: int = 5,
) -> list[tuple[int, int]]:
    """Generate ten plausible score lines for the manual demo mode."""
    generator = rng or random
    return [
        (generator.randint(0, max_goals), generator.randint(0, max_goals))
        for _ in range(HISTORY_LENGTH)
    ]


def summarize_history(
    history: list[tuple[int, int] | None],
) -> TeamHistorySummary:
    """Summarize results and recent form without moving missing matches."""
    known_matches = [score for score in history if score is not None]
    wins = sum(goals > conceded for goals, conceded in known_matches)
    draws = sum(goals == conceded for goals, conceded in known_matches)
    losses = sum(goals < conceded for goals, conceded in known_matches)
    goals_for = sum(goals for goals, _ in known_matches)
    goals_against = sum(conceded for _, conceded in known_matches)
    points = wins * 3 + draws

    recent_matches = [score for score in history[:5] if score is not None]
    recent_points = sum(
        3 if goals > conceded else 1 if goals == conceded else 0
        for goals, conceded in recent_matches
    )
    recent_points_per_match = (
        recent_points / len(recent_matches) if recent_matches else None
    )
    if recent_points_per_match is None:
        form_level = "Non disponible"
    elif recent_points_per_match >= 2:
        form_level = "Bonne"
    elif recent_points_per_match >= 1:
        form_level = "Moyenne"
    else:
        form_level = "Faible"

    return TeamHistorySummary(
        matches=len(known_matches),
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for=goals_for,
        goals_against=goals_against,
        points=points,
        goal_difference=goals_for - goals_against,
        recent_points_per_match=recent_points_per_match,
        form_level=form_level,
    )


def history_from_match(
    match: Mapping[str, object], team: str
) -> list[tuple[int, int] | None]:
    """Extract one team's ordered n-1 to n-10 score history from a row."""
    if team not in {"home", "away"}:
        raise ValueError("L'équipe historique doit être 'home' ou 'away'.")

    history: list[tuple[int, int] | None] = []
    for match_number in range(1, HISTORY_LENGTH + 1):
        goals = match[f"{team}_team_history_goal_{match_number}"]
        conceded = match[f"{team}_team_history_opponent_goal_{match_number}"]
        if pd.isna(goals) or pd.isna(conceded):
            history.append(None)
            continue
        try:
            numeric_goals = float(goals)
            numeric_conceded = float(conceded)
        except (TypeError, ValueError) as error:
            raise ValueError("Un score historique n'est pas numérique.") from error
        if numeric_goals < 0 or numeric_conceded < 0:
            raise ValueError("Les scores historiques ne peuvent pas être négatifs.")
        if not numeric_goals.is_integer() or not numeric_conceded.is_integer():
            raise ValueError("Les scores historiques doivent être des nombres entiers.")
        history.append((int(numeric_goals), int(numeric_conceded)))
    return history


def _make_form_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in HISTORY_FEATURES if column not in matches.columns]
    if missing:
        raise ValueError(
            "Le fichier ne contient pas toutes les colonnes requises. "
            f"Colonnes manquantes : {', '.join(missing)}"
        )

    features = pd.DataFrame(index=matches.index)
    scores: dict[str, pd.Series] = {}
    conceded_scores: dict[str, pd.Series] = {}
    for team in ("home", "away"):
        for match_number in range(1, HISTORY_LENGTH + 1):
            score_column = f"{team}_team_history_goal_{match_number}"
            conceded_column = f"{team}_team_history_opponent_goal_{match_number}"
            scores[score_column] = pd.to_numeric(matches[score_column], errors="coerce")
            conceded_scores[conceded_column] = pd.to_numeric(
                matches[conceded_column], errors="coerce"
            )
            for column, values in (
                (score_column, scores[score_column]),
                (conceded_column, conceded_scores[conceded_column]),
            ):
                invalid_values = matches[column].notna() & values.isna()
                if invalid_values.any():
                    raise ValueError(
                        f"La colonne {column} contient des scores non numériques."
                    )
                if values.lt(0).any():
                    raise ValueError("Les scores historiques ne peuvent pas être négatifs.")

    for team in ("home", "away"):
        for start, end, suffix in HISTORY_WINDOWS:
            points = pd.Series(0.0, index=matches.index)
            match_count = pd.Series(0, index=matches.index)
            for match_number in range(start, end + 1):
                goals = scores[f"{team}_team_history_goal_{match_number}"]
                conceded = conceded_scores[
                    f"{team}_team_history_opponent_goal_{match_number}"
                ]
                valid = goals.notna() & conceded.notna()
                match_points = goals.gt(conceded).astype(float) * 3
                match_points += goals.eq(conceded).astype(float)
                points += match_points.where(valid, 0.0)
                match_count += valid.astype(int)

            features[f"{team}_{suffix}"] = points
            features[f"{team}_{suffix}_match_count"] = match_count

    usable = pd.Series(True, index=matches.index)
    for team in ("home", "away"):
        for _, _, suffix in HISTORY_WINDOWS:
            usable &= features[f"{team}_{suffix}_match_count"].ge(
                MIN_MATCHES_PER_WINDOW
            )

    return features.loc[:, MODEL_FEATURES], usable


def _insufficient_history_error(matches: pd.DataFrame, usable: pd.Series) -> ValueError:
    rows = matches.index[~usable].tolist()
    preview = ", ".join(str(row) for row in rows[:10])
    more = "…" if len(rows) > 10 else ""
    return ValueError(
        "Chaque équipe doit avoir au moins 3 matchs valides dans chacune "
        "des fenêtres 1–5 et 6–10. "
        f"Lignes concernées : {preview}{more}"
    )


def prepare_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate the four point totals used by the notebook's classifier."""
    features, usable = _make_form_features(matches)
    if not usable.all():
        raise _insufficient_history_error(matches, usable)
    return features


def train_model(
    training_data_path: str | Path,
    before_date: str | pd.Timestamp | None = None,
) -> FormModel:
    """Fit the form model, optionally using only matches before a cutoff.

    When ``before_date`` is given, only matches strictly before that date are used.
    """
    training_data_path = Path(training_data_path)
    if not training_data_path.is_file():
        raise FileNotFoundError(
            f"Fichier d'entraînement introuvable : {training_data_path}"
        )

    selected_columns = [*HISTORY_FEATURES, "target"]
    if before_date is not None:
        selected_columns.append("match_date")
    training_data = pd.read_csv(training_data_path, usecols=selected_columns)
    if before_date is not None:
        cutoff = pd.Timestamp(before_date)
        match_dates = pd.to_datetime(training_data["match_date"], errors="coerce")
        if match_dates.isna().any():
            raise ValueError("La colonne match_date contient des dates invalides.")
        training_data = training_data.loc[match_dates.lt(cutoff)]
        if training_data.empty:
            raise ValueError(
                "Aucun match d'entraînement n'est antérieur à la date de coupure."
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

    features, usable = _make_form_features(training_data)
    if not usable.any():
        raise ValueError(
            "Aucun match ne possède au moins 3 historiques valides dans "
            "chacune des fenêtres 1–5 et 6–10."
        )

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(features.loc[usable], labels.loc[usable])
    outcome_prior = (
        labels.loc[usable]
        .value_counts(normalize=True)
        .reindex(OUTCOMES, fill_value=0.0)
        .to_dict()
    )
    return FormModel(classifier=classifier, outcome_prior=outcome_prior)


def make_match_features(
    home_history: list[tuple[int, int] | None],
    away_history: list[tuple[int, int] | None],
) -> pd.DataFrame:
    """Build one row from up to ten previous (goals, goals_conceded) scores."""
    row: dict[str, float] = {column: float("nan") for column in HISTORY_FEATURES}
    for team, history in (("home", home_history), ("away", away_history)):
        if len(history) > HISTORY_LENGTH:
            raise ValueError(
                f"Au maximum {HISTORY_LENGTH} matchs récents peuvent être saisis."
            )
        for match_number, score in enumerate(history, start=1):
            if score is None:
                continue
            goals, opponent_goals = score
            if goals < 0 or opponent_goals < 0:
                raise ValueError("Les scores ne peuvent pas être négatifs.")
            row[f"{team}_team_history_goal_{match_number}"] = float(goals)
            row[f"{team}_team_history_opponent_goal_{match_number}"] = float(
                opponent_goals
            )

    match_features = pd.DataFrame([row], columns=HISTORY_FEATURES)
    prepare_features(match_features)
    return match_features


def predict_probabilities(
    model: FormModel,
    matches: pd.DataFrame,
    allow_incomplete_history: bool = False,
) -> pd.DataFrame:
    """Return home/draw/away probabilities in a stable column order."""
    features, usable = _make_form_features(matches)
    if not allow_incomplete_history and not usable.all():
        raise _insufficient_history_error(matches, usable)

    probabilities = pd.DataFrame(
        [model.outcome_prior] * len(matches), index=matches.index, columns=OUTCOMES
    )
    if usable.any():
        predicted = model.classifier.predict_proba(features.loc[usable])
        for index, outcome in enumerate(model.classifier.classes_):
            probabilities.loc[usable, outcome] = predicted[:, index]
    return probabilities


def count_matches_with_sufficient_history(matches: pd.DataFrame) -> int:
    """Count rows meeting the notebook's minimum-history rule."""
    _, usable = _make_form_features(matches)
    return int(usable.sum())


def matches_with_sufficient_history(matches: pd.DataFrame) -> pd.Series:
    """Return the eligibility mask used by training and prediction."""
    _, usable = _make_form_features(matches)
    return usable


