"""
Data cleaning utility functions.

This module provides reusable functions for cleaning raw datasets, handling
missing values, correcting inconsistent data formats, removing duplicates,
and preparing data for downstream preprocessing and modelling.
"""

from __future__ import annotations

import pandas as pd


def height_to_inches(height: pd.Series) -> pd.Series:
    """Convert basketball height values to numeric height in inches.

    The input may contain standard feet-inches representations such as
    ``6-7`` or ``6'7``, as well as month-name representations such as
    ``6-Jul`` that can occur when height values have been interpreted as
    dates. Invalid or non-height values are returned as missing values.

    Args:
        height: Series containing player height values.

    Returns:
        A float Series containing height values converted to inches.
        Values that cannot be interpreted as valid feet-inches heights
        are returned as missing values.
    """
    # Convert values to strings and remove surrounding whitespace.
    height_clean = height.astype("string").str.strip()

    # Convert month abbreviations back to their corresponding
    # numerical inch values (e.g. "6-Jul" -> "6-7").
    month_to_number = {
        "Jan": "1",
        "Feb": "2",
        "Mar": "3",
        "Apr": "4",
        "May": "5",
        "Jun": "6",
        "Jul": "7",
        "Aug": "8",
        "Sep": "9",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }

    for month, number in month_to_number.items():
        height_clean = height_clean.str.replace(
            rf"^(?P<feet>\d+)-{month}$",
            rf"\g<feet>-{number}",
            regex=True,
        )

    # Convert apostrophe-separated heights such as 6'4 to 6-4.
    height_clean = height_clean.str.replace(
        r"^(\d+)'(\d+)$",
        r"\1-\2",
        regex=True,
    )

    # Extract the feet and inches components from valid heights.
    height_parts = height_clean.str.extract(
        r"^(?P<feet>\d+)-(?P<inches>\d+)$"
    )

    # Convert the extracted components to numeric values.
    feet = pd.to_numeric(
        height_parts["feet"],
        errors="coerce",
    )

    inches = pd.to_numeric(
        height_parts["inches"],
        errors="coerce",
    )

    # Convert feet-inches into total inches.
    height_inches = (feet * 12) + inches

    # Reject physically invalid height representations.
    height_inches = height_inches.where(
        (feet >= 4)
        & (feet <= 8)
        & (inches >= 0)
        & (inches <= 11)
    )

    return height_inches.rename("height_inches")

def remove_features(
    df: pd.DataFrame,
    columns_to_remove: list[str],
) -> pd.DataFrame:
    """
    Remove specified features from a DataFrame.

    This function removes the columns provided in ``columns_to_remove``
    while leaving the original DataFrame unchanged.

    Args:
        df (pd.DataFrame):
            Input DataFrame from which features will be removed.

        columns_to_remove (list[str]):
            List of column names to remove.

    Returns:
        pd.DataFrame:
            A copy of the input DataFrame with the specified columns removed.

    Raises:
        TypeError:
            If ``columns_to_remove`` is not a list of strings.
    """
    if not isinstance(columns_to_remove, list):
        raise TypeError("columns_to_remove must be a list of strings.")

    if not all(isinstance(column, str) for column in columns_to_remove):
        raise TypeError("All values in columns_to_remove must be strings.")

    return df.drop(
        columns=columns_to_remove,
        errors="ignore",
    ).copy()


def calculate_age(
    dob: pd.Series,
    season_year: pd.Series,
) -> pd.Series:
    """
    Calculate player age from date of birth and season year.

    Age is calculated as the season year minus the year of birth.
    Missing or invalid dates result in missing age values.

    Args:
        dob (pd.Series):
            Series containing player dates of birth.

        season_year (pd.Series):
            Series containing the season year corresponding to each player
            record.

    Returns:
        pd.Series:
            A numerical Series containing the player's age in years.

    Raises:
        ValueError:
            If the input Series do not have the same length.
    """
    if len(dob) != len(season_year):
        raise ValueError(
            "dob and season_year must have the same length."
        )

    birth_dates = pd.to_datetime(
        dob,
        errors="coerce",
    )

    season_year_numeric = pd.to_numeric(
        season_year,
        errors="coerce",
    )

    age = season_year_numeric - birth_dates.dt.year

    return age.rename("age")