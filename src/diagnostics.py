# src/diagnostics.py
#
# A script to diagnose the composition of factor portfolios over time.
#
import matplotlib.pyplot as plt
import polars as pl

from config import settings
from src.runner import get_data_filepath


def run_diagnostics(characteristic: str, weighting_scheme: str = "ew"):
    """
    Analyzes and plots the composition of factor portfolios over time.

    Args:
        characteristic (str): The characteristic to analyze (e.g., 'ret_12_1').
        weighting_scheme (str): The weighting scheme to report on.
    """
    print(f"--- Running Diagnostics for: {characteristic} ({weighting_scheme.upper()}) ---")

    # --- This logic is copied from the start of construct_factor_portfolio ---
    char_data_file = get_data_filepath()
    required_cols = ["eom", "permno", "crsp_exchcd", "me", characteristic]
    all_cols = list(dict.fromkeys(required_cols))

    char_df = pl.read_parquet(char_data_file, columns=all_cols).drop_nulls(
        subset=[characteristic, "me"]
    )

    if characteristic == "be_me":
        char_df = char_df.filter(pl.col(characteristic) > 0)

    nyse_stocks = char_df.filter(pl.col("crsp_exchcd") == 1)

    size_breakpoints = (
        nyse_stocks.group_by("eom")
        .agg(
            pl.col("me").quantile(0.20).alias("me_p20"),
        )
        .sort("eom")
    )

    char_df = char_df.join(size_breakpoints, on="eom")

    non_micro_caps = char_df.filter(pl.col("me") > pl.col("me_p20"))
    char_breakpoints = (
        non_micro_caps.group_by("eom")
        .agg(
            pl.col(characteristic).quantile(1 / 3).alias("char_p33"),
            pl.col(characteristic).quantile(2 / 3).alias("char_p67"),
        )
        .sort("eom")
    )

    char_df = char_df.join(char_breakpoints, on="eom")

    char_df = char_df.with_columns(
        pl.when(pl.col(characteristic) <= pl.col("char_p33"))
        .then(pl.lit("Low"))
        .when(pl.col(characteristic) > pl.col("char_p67"))
        .then(pl.lit("High"))
        .otherwise(pl.lit("Mid"))
        .alias("portfolio")
    )
    # --- End of copied logic ---

    # --- Calculate time-series statistics for each portfolio ---
    diagnostics_df = (
        char_df.group_by(["eom", "portfolio"])
        .agg(
            pl.len().alias("n_stocks"),
            pl.mean("me").alias("avg_me_usd_mil"),
            pl.mean(characteristic).alias(f"avg_{characteristic}"),
        )
        .sort("eom")
        .to_pandas()
    )

    # --- Plot the results ---
    metrics = ["n_stocks", "avg_me_usd_mil", f"avg_{characteristic}"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))
        for port_name in ["Low", "Mid", "High"]:
            subset = diagnostics_df[diagnostics_df["portfolio"] == port_name]
            ax.plot(subset["eom"], subset[metric], label=port_name)

        ax.set_title(
            f"Time Series of {metric.replace('_', ' ').title()} for '{characteristic}' Portfolios",
            fontsize=16,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, which="both", linestyle="--")

        # Save the plot
        output_dir = settings.PROJECT_ROOT / "plots" / "diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{characteristic}_{metric}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Diagnostic plot saved to: {plot_path}")


if __name__ == "__main__":
    # Run diagnostics for the Momentum factor, which you noted had a crash in the 2000s.
    run_diagnostics(characteristic="ret_12_1")
