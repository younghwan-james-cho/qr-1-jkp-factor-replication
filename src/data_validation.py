"""Fail-closed structural checks; these do not certify the research methodology."""

import polars as pl


def validate_raw_data(df: pl.DataFrame) -> None:
    required = {"eom", "id", "me", "ret_exc_lead1m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.is_empty():
        raise ValueError("Raw data is empty")
    if any(df[col].null_count() for col in ("eom", "id")):
        raise ValueError("Null observation keys")
    if df.select(pl.struct("eom", "id").is_duplicated().any()).item():
        raise ValueError("Duplicate (eom, id) observations")
    if df.schema["eom"] != pl.Date:
        raise ValueError("eom must be a Date")
    if df.select((pl.col("eom") != pl.col("eom").dt.month_end()).any()).item():
        raise ValueError("eom must contain month-end dates")
    for column in ("me", "ret_exc_lead1m"):
        if not df.schema[column].is_numeric():
            raise ValueError(f"{column} must be numeric")
        if df.select((pl.col(column).is_not_null() & ~pl.col(column).is_finite()).any()).item():
            raise ValueError(f"Non-finite {column}")
    if df.select((pl.col("me") <= 0).any()).item():
        raise ValueError("Market equity must be positive when observed")
