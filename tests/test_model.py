import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model import (
    HISTORY_FEATURES,
    MODEL_FEATURES,
    OUTCOMES,
    count_matches_with_sufficient_history,
    make_match_features,
    predict_probabilities,
    prepare_features,
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


if __name__ == "__main__":
    unittest.main()
