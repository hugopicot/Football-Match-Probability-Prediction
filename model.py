"""Training and prediction helpers for the notebook's form-based model."""

from dataclasses import dataclass
from pathlib import Path

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


def train_model(training_data_path: str | Path) -> FormModel:
    """Fit the notebook's logistic regression on rows with enough history."""
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
