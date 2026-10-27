import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model import (
    HISTORY_FEATURES,
    HISTORY_LENGTH,
    MODEL_FEATURES,
    OUTCOMES,
    count_matches_with_sufficient_history,
    generate_random_history,
    history_from_match,
    make_match_features,
    matches_with_sufficient_history,
    predict_probabilities,
    prepare_features,
    summarize_history,
    train_model,
)


def make_history_row(
    home_history: list[tuple[int, int] | None],
    away_history: list[tuple[int, int] | None],
) -> dict[str, float]:
    row = {column: float("nan") for column in HISTORY_FEATURES}
    for team, history in (("home", home_history), ("away", away_history)):
        for number, score in enumerate(history, start=1):
            if score is not None:
                row[f"{team}_team_history_goal_{number}"] = score[0]
                row[f"{team}_team_history_opponent_goal_{number}"] = score[1]
    return row


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home_history = [
            (2, 0), (0, 0), (0, 1), (1, 0), (0, 0),
            (0, 1), (1, 1), (3, 2), (1, 0), (0, 2),
        ]
        self.away_history = [
            (0, 2), (0, 0), (1, 0), (1, 1), (0, 1),
            (2, 0), (1, 2), (0, 0), (0, 1), (2, 1),
        ]
        self.matches = pd.DataFrame(
            [make_history_row(self.home_history, self.away_history)]
        )

    def test_prepare_features_sums_points_in_both_windows(self) -> None:
        features = prepare_features(self.matches)

        self.assertEqual(tuple(features.columns), MODEL_FEATURES)
        self.assertEqual(
            features.iloc[0].to_dict(),
            {"home_5": 8.0, "away_5": 5.0, "home_610": 7.0, "away_610": 7.0},
        )

    def test_prepare_features_rejects_insufficient_window_history(self) -> None:
        history = self.home_history.copy()
        history[0] = None
        history[1] = None
        history[2] = None
        matches = pd.DataFrame(
            [make_history_row(history, self.away_history)], index=["match-1"]
        )

        with self.assertRaisesRegex(ValueError, "au moins 3 matchs valides"):
            prepare_features(matches)

    def test_make_match_features_preserves_missing_match_positions(self) -> None:
        history = self.home_history.copy()
        history[1] = None

        features = make_match_features(history, self.away_history)
        self.assertTrue(
            pd.isna(features.loc[0, "home_team_history_goal_2"])
        )
        self.assertEqual(prepare_features(features).loc[0, "home_5"], 7.0)

    def test_training_and_prediction_return_all_outcomes(self) -> None:
        rows = []
        for outcome in OUTCOMES:
            for _ in range(2):
                row = make_history_row(self.home_history, self.away_history)
                row["target"] = outcome
                rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            training_path = Path(directory) / "train.csv"
            pd.DataFrame(rows).to_csv(training_path, index=False)
            classifier = train_model(training_path)

        predictions = predict_probabilities(
            classifier,
            make_match_features(self.home_history, self.away_history),
        )
        self.assertEqual(tuple(predictions.columns), OUTCOMES)
        self.assertAlmostEqual(float(predictions.iloc[0].sum()), 1.0)

        incomplete_away_history = self.away_history[:5] + [None] * 5
        incomplete_match = pd.DataFrame(
            [make_history_row(self.home_history, incomplete_away_history)]
        )
        self.assertEqual(count_matches_with_sufficient_history(incomplete_match), 0)
        fallback = predict_probabilities(
            classifier, incomplete_match, allow_incomplete_history=True
        )
        self.assertEqual(fallback.iloc[0].to_dict(), classifier.outcome_prior)
        with self.assertRaisesRegex(ValueError, "au moins 3 matchs valides"):
            predict_probabilities(classifier, incomplete_match)

    def test_history_from_match_reads_recent_first_and_missing_as_none(self) -> None:
        row = pd.Series(make_history_row(self.home_history, [None] * 10))

        history = history_from_match(row, "home")
        self.assertEqual(history, self.home_history)
        self.assertEqual(history_from_match(row, "away"), [None] * 10)

    def test_matches_with_sufficient_history_matches_count(self) -> None:
        complete = make_history_row(self.home_history, self.away_history)
        incomplete = make_history_row(self.home_history, self.away_history[:5] + [None] * 5)
        matches = pd.DataFrame([complete, incomplete])

        mask = matches_with_sufficient_history(matches)
        self.assertEqual(mask.tolist(), [True, False])
        self.assertEqual(int(mask.sum()), count_matches_with_sufficient_history(matches))

    def test_generate_random_history_returns_ten_valid_scores(self) -> None:
        history = generate_random_history(max_goals=4)

        self.assertEqual(len(history), HISTORY_LENGTH)
        for goals, conceded in history:
            self.assertTrue(0 <= goals <= 4)
            self.assertTrue(0 <= conceded <= 4)

    def test_summarize_history_counts_results_and_recent_form(self) -> None:
        history = [(2, 0), (1, 1), (0, 3), None, (4, 2)] + [None] * 5

        summary = summarize_history(history)
        self.assertEqual(summary.matches, 4)
        self.assertEqual((summary.wins, summary.draws, summary.losses), (2, 1, 1))
        self.assertEqual(summary.goals_for, 7)
        self.assertEqual(summary.goals_against, 6)
        self.assertEqual(summary.points, 7)
        self.assertEqual(summary.goal_difference, 1)
        self.assertAlmostEqual(summary.recent_points_per_match, 7 / 4)
        self.assertEqual(summary.form_level, "Correcte")

        empty = summarize_history([None] * 10)
        self.assertEqual(empty.matches, 0)
        self.assertIsNone(empty.recent_points_per_match)
        self.assertEqual(empty.form_level, "Aucune donnée")

    def test_train_model_before_date_ignores_later_matches(self) -> None:
        early_home = make_history_row(self.home_history, self.away_history)
        early_home.update({"target": "home", "match_date": "2020-01-01 10:00:00"})
        early_draw = make_history_row(self.home_history, self.away_history)
        early_draw.update({"target": "draw", "match_date": "2020-02-01 10:00:00"})
        late_away = make_history_row(self.away_history, self.home_history)
        late_away.update({"target": "away", "match_date": "2021-01-01 10:00:00"})

        with tempfile.TemporaryDirectory() as directory:
            training_path = Path(directory) / "train.csv"
            pd.DataFrame([early_home, early_draw, late_away]).to_csv(
                training_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "au moins 3 historiques valides"):
                train_model(training_path, before_date="2019-06-01")
            model = train_model(training_path, before_date="2020-06-01")

        self.assertEqual(sorted(model.classifier.classes_.tolist()), ["draw", "home"])


if __name__ == "__main__":
    unittest.main()
