# src/portfolio_construction.py
#
# Functions for constructing long-short factor portfolios.
#

from pathlib import Path

import polars as pl

from src.data_validation import validate_raw_data


def construct_factor_portfolio(
    char_data_path: Path,
    characteristic: str,
    factor_name: str,
    weighting_scheme: str = "vw_cap",
    long_short_direction: str = "long",
) -> pl.DataFrame:
    """
    Constructs a long-short factor portfolio based on a single characteristic,
    supporting multiple weighting schemes and return winsorization.
    """
    if weighting_scheme not in {"ew", "vw", "vw_cap"}:
        raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")
    if long_short_direction not in {"long", "short"}:
        raise ValueError(f"Unknown long-short direction: {long_short_direction}")
    if factor_name == "eom":
        raise ValueError("factor_name cannot replace the date key")
    base_cols = [
        "eom",
        "id",
        "permno",
        "crsp_exchcd",
        "source_crsp",
        "size_grp",
        "me",
        "ret_exc_lead1m",
    ]
    all_cols = base_cols + [characteristic]
    required_cols = list(dict.fromkeys(all_cols))

    # Add stricter initial screen to match the original R code.
    raw = pl.read_parquet(char_data_path, columns=required_cols)
    validate_raw_data(raw)
    if not raw.schema[characteristic].is_numeric():
        raise ValueError("Characteristic must be numeric")
    if raw.select(
        (pl.col(characteristic).is_not_null() & ~pl.col(characteristic).is_finite()).any()
    ).item():
        raise ValueError("Non-finite characteristic")
    char_df = raw.drop_nulls(subset=[characteristic, "me", "size_grp", "ret_exc_lead1m"])

    if char_df.is_empty():
        raise ValueError("No eligible observations after screening")
    if char_df.filter(
        ~pl.col("source_crsp").is_in([0, 1]) | pl.col("source_crsp").is_null()
    ).height:
        raise ValueError("source_crsp must be 0 or 1")

    # --- Implement Return Winsorization ---
    # Per the JKP R code, winsorize non-CRSP returns based on CRSP cutoffs.

    # 1. Calculate the 0.1% and 99.9% return cutoffs for each month using only CRSP stocks.
    crsp_ret_cutoffs = (
        char_df.filter(pl.col("source_crsp") == 1)
        .group_by("eom")
        .agg(
            pl.col("ret_exc_lead1m").quantile(0.001, interpolation="nearest").alias("ret_p001"),
            pl.col("ret_exc_lead1m").quantile(0.999, interpolation="nearest").alias("ret_p999"),
        )
    )

    # 2. Join the cutoffs to the main dataframe.
    char_df = char_df.join(crsp_ret_cutoffs, on="eom", how="left")

    if char_df.filter((pl.col("source_crsp") == 0) & pl.col("ret_p001").is_null()).height:
        raise ValueError("Missing CRSP winsorization reference for non-CRSP observations")

    # 3. Apply the winsorization only to Compustat-sourced stocks (source_crsp == 0).
    char_df = char_df.with_columns(
        pl.when(pl.col("source_crsp") == 0)
        .then(
            pl.col("ret_exc_lead1m").clip(
                lower_bound=pl.col("ret_p001"), upper_bound=pl.col("ret_p999")
            )
        )
        .otherwise(pl.col("ret_exc_lead1m"))
        .alias("ret_exc_lead1m")
    ).drop(["ret_p001", "ret_p999"])  # Clean up cutoff columns.
    # --- End of Winsorization Logic ---

    if characteristic == "be_me":
        char_df = char_df.filter(pl.col(characteristic) > 0)

    nyse_stocks = char_df.filter(pl.col("crsp_exchcd") == 1)

    size_breakpoints = (
        nyse_stocks.group_by("eom")
        .agg(
            pl.col("me").quantile(0.20, interpolation="nearest").alias("me_p20"),
            pl.col("me").quantile(0.80, interpolation="nearest").alias("me_p80"),
        )
        .sort("eom")
    )

    char_df = char_df.join(size_breakpoints, on="eom", how="left")

    # Drop rows that could not be assigned a breakpoint (e.g., in months with no NYSE stocks)
    char_df = char_df.drop_nulls(subset=["me_p20"])

    non_micro_caps = char_df.filter(pl.col("me") > pl.col("me_p20"))
    char_breakpoints = (
        non_micro_caps.group_by("eom")
        .agg(
            pl.col(characteristic).quantile(1 / 3, interpolation="nearest").alias("char_p33"),
            pl.col(characteristic).quantile(2 / 3, interpolation="nearest").alias("char_p67"),
        )
        .sort("eom")
    )
    char_df = char_df.join(char_breakpoints, on="eom", how="left")

    # Drop rows that could not be assigned a characteristic breakpoint
    char_df = char_df.drop_nulls(subset=["char_p33"])

    char_df = char_df.with_columns(
        pl.when(pl.col(characteristic) <= pl.col("char_p33"))
        .then(pl.lit("Low"))
        .when(pl.col(characteristic) > pl.col("char_p67"))
        .then(pl.lit("High"))
        .otherwise(pl.lit("Mid"))
        .alias("portfolio")
    )

    if weighting_scheme == "ew":
        weighting_expr = pl.mean("ret_exc_lead1m").alias("port_ret")
    else:
        weights = pl.col("me")
        if weighting_scheme == "vw_cap":
            weights = pl.min_horizontal("me", "me_p80")
        # A common positive scale leaves normalized weights unchanged and avoids
        # overflow when a valid market-equity unit makes the raw sum too large.
        scaled_weights = weights / weights.max()
        weighting_expr = (
            ((scaled_weights / scaled_weights.sum()) * pl.col("ret_exc_lead1m"))
            .sum()
            .alias("port_ret")
        )

    portfolio_returns = (
        char_df.group_by(["eom", "portfolio"])
        .agg(weighting_expr, pl.len().alias("n_stocks"))
        .sort("eom")
    )

    if not portfolio_returns["port_ret"].is_finite().all():
        raise ValueError("Non-finite portfolio returns")

    portfolio_returns_wide = portfolio_returns.pivot(
        index="eom", on="portfolio", values=["port_ret", "n_stocks"]
    ).sort("eom")
    if not {"n_stocks_Low", "n_stocks_High"} <= set(portfolio_returns_wide.columns):
        raise ValueError("Missing long or short portfolio leg")
    portfolio_returns_wide = portfolio_returns_wide.filter(
        (pl.col("n_stocks_Low") >= 5) & (pl.col("n_stocks_High") >= 5)
    )

    if portfolio_returns_wide.is_empty():
        raise ValueError("No months have at least five stocks in both legs")

    if long_short_direction == "long":
        factor_returns = portfolio_returns_wide.with_columns(
            (pl.col("port_ret_High") - pl.col("port_ret_Low")).alias(factor_name)
        )
    else:
        factor_returns = portfolio_returns_wide.with_columns(
            (pl.col("port_ret_Low") - pl.col("port_ret_High")).alias(factor_name)
        )

    return factor_returns.select(["eom", factor_name])
