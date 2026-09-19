# src/data_ingestion.py
#
# Functions for downloading firm-level data from WRDS.
#

import datetime
import logging
import re
import sys
from pathlib import Path

import polars as pl
import wrds

# Import settings from our central config files
from config import settings

try:
    from config.wrds_config import WRDS_USERNAME
except ImportError:
    WRDS_USERNAME = None


def setup_logger():
    """Set up a basic logger to show INFO level messages."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def download_jkp_char_data(
    characteristics: list[str],
    output_path: Path,
    wrds_username: str | None = WRDS_USERNAME,
    start_date: str | None = None,
    country: str = "USA",
    filter_common: bool = True,
    filter_exch_main: bool = True,
    filter_primary_sec: bool = True,
    filter_obs_main: bool = True,
) -> None:
    """Connects to WRDS and downloads specified firm characteristics data."""
    logger.info("--- Starting JKP Characteristic Data Download ---")

    if not wrds_username or wrds_username == "your_username_here":
        error_msg = (
            "WRDS_USERNAME not set. Please update config/wrds_config.py or pass it as an argument."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite input: {output_path}")
    if not characteristics or any(not re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in characteristics):
        raise ValueError("Characteristics must be nonempty SQL identifiers")
    if not re.fullmatch(r"[A-Z]{3}", country):
        raise ValueError("Country must be a three-letter uppercase code")
    if start_date is not None:
        datetime.date.fromisoformat(start_date)
    db = None
    try:
        logger.info(f"Connecting to WRDS with username: {wrds_username}...")
        db = wrds.Connection(wrds_username=wrds_username)
        logger.info("Successfully connected to WRDS.")

        # Add 'source_crsp' and 'size_grp' for winsorization and screening.
        core_cols = [
            "eom",
            "id",
            "permno",
            "crsp_exchcd",
            "source_crsp",
            "size_grp",
            "me",
            "ret_exc_lead1m",
        ]

        unique_chars = sorted(list(set(characteristics)))
        all_cols = list(dict.fromkeys(core_cols + unique_chars))
        columns_str = ", ".join(f'"{c}"' for c in all_cols)

        filters = ["excntry = %(country)s"]
        params = {"country": country}
        if filter_common:
            filters.append("common = 1")
        if filter_exch_main:
            filters.append("exch_main = 1")
        if filter_primary_sec:
            filters.append("primary_sec = 1")
        if filter_obs_main:
            filters.append("obs_main = 1")
        if start_date:
            filters.append("eom >= %(start_date)s")
            params["start_date"] = start_date
        filters_str = "\n            AND ".join(filters)

        query = f"SELECT {columns_str} FROM contrib.global_factor WHERE {filters_str}"
        logger.info(f"Executing query for characteristics: {unique_chars}")

        pandas_df = db.raw_sql(query, params=params, date_cols=["eom"])
        logger.info(
            "Query executed successfully. Fetched %d rows and %d columns.",
            pandas_df.shape[0],
            pandas_df.shape[1],
        )

    except Exception:
        logger.error("An error occurred during WRDS data download.", exc_info=True)
        raise
    finally:
        if db:
            db.close()
            logger.info("WRDS connection closed.")

    try:
        from src.data_validation import validate_raw_data

        polars_df = pl.from_pandas(pandas_df).with_columns(pl.col("eom").cast(pl.Date))
        validate_raw_data(polars_df)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as stream:
            polars_df.write_parquet(stream)
        logger.info("Data saved successfully to %s. Shape: %s", output_path, polars_df.shape)
    except Exception:
        logger.error("Failed to convert or save data.", exc_info=True)
        raise

    logger.info("--- Characteristic Data Download Complete ---")


if __name__ == "__main__":
    end_date_str = datetime.date.today().strftime("%Y-%m-%d")
    start_fname = datetime.datetime.strptime(settings.START_DATE_STR, "%Y-%m-%d").strftime("%Y%m")
    end_fname = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%Y%m")
    output_filename = f"jkp_char_usa_{start_fname}_{end_fname}.parquet"
    output_file_path = settings.PROJECT_ROOT / "data" / "raw" / "char" / output_filename

    # Extract the unique characteristics from the full list of factors.
    characteristics_list = list(set(char for char, _, _ in settings.FACTORS_TO_REPLICATE))

    try:
        download_jkp_char_data(
            characteristics=characteristics_list,
            output_path=output_file_path,
            start_date=settings.START_DATE_STR,
        )
    except Exception:
        logger.critical("Script execution failed. Please check the logs above for details.")
        sys.exit(1)
