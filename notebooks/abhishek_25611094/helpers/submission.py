"""
Kaggle submission utility functions.

This module contains reusable functions for generating prediction files in
the required competition submission format. The functions ensure prediction
outputs are correctly aligned with player identifiers and exported in a
reproducible manner.
"""

from __future__ import annotations
import pandas as pd

def create_submission_file(
    pid,
    year,
    predicted_probabilities,
    output_path=None,
):
    """
    Create a player-level Kaggle submission by selecting
    the most recent available season for each player.

    Multiple test observations can belong to the same player
    because players may appear across multiple seasons.
    """

    predictions = pd.DataFrame({
        "pid": pid,
        "year": year,
        "prediction": predicted_probabilities,
    })

    # Sort so that the most recent season for each player
    # appears last.
    predictions = predictions.sort_values(
        ["pid", "year"]
    )

    # Keep the most recent available season for each player.
    submission = (
        predictions
        .drop_duplicates(
            subset="pid",
            keep="last",
        )
        [["pid", "prediction"]]
        .rename(columns={"prediction": "label"})
    )

    # Basic validation.
    if submission["pid"].isna().any():
        raise ValueError("Submission contains missing PIDs.")

    if submission["label"].isna().any():
        raise ValueError("Submission contains missing predictions.")

    if not submission["label"].between(0, 1).all():
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    if submission["pid"].duplicated().any():
        raise ValueError(
            "Submission still contains duplicate PIDs."
        )

    # Save the submission if an output path is provided.
    if output_path is not None:
        submission.to_csv(
            output_path,
            index=False,
        )

    return submission