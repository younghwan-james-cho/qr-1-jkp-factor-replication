# src/validation.py
#
# Functions for validating replicated factor returns.
#

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_cumulative_returns(df: pl.DataFrame, factor_name: str, plot_subdir: str):
    """
    Generates and saves a plot of cumulative factor returns into a specific subdirectory.
    """
    plot_df = df.to_pandas()

    plot_df["replicated_cumulative"] = (1 + plot_df[factor_name]).cumprod()
    plot_df["benchmark_cumulative"] = (1 + plot_df["benchmark_ret"]).cumprod()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        plot_df["eom"], plot_df["replicated_cumulative"], label="Replicated Factor", linewidth=2
    )
    ax.plot(
        plot_df["eom"],
        plot_df["benchmark_cumulative"],
        label="Benchmark Factor",
        linestyle="--",
        linewidth=2,
    )

    ax.set_title(f"Cumulative Performance: {factor_name}", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.legend(fontsize=12)
    ax.set_yscale("log")

    fig.tight_layout()

    output_dir = Path("plots") / plot_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_filename = f"{factor_name.lower()}.png"
    plot_path = output_dir / plot_filename

    fig.savefig(plot_path)
    print(f"Performance plot saved to: {plot_path}")
    plt.close(fig)


def align_monthly_returns(replicated: pl.DataFrame, benchmark: pl.DataFrame) -> pl.DataFrame:
    """Map formation month t to realized month-end t+1, then join one-to-one."""
    if len(replicated.columns) != 2 or "eom" not in replicated.columns:
        raise ValueError("Replicated returns need eom and exactly one return column")
    factor = next(col for col in replicated.columns if col != "eom")
    if factor == "benchmark_ret":
        raise ValueError("Reserved return column: benchmark_ret")
    for frame, column in ((replicated, factor), (benchmark, "benchmark_ret")):
        if frame.is_empty():
            raise ValueError("Empty return series")
        if frame.schema["eom"] != pl.Date:
            raise ValueError("Return dates must be Date values")
        if frame["eom"].null_count() or frame["eom"].is_duplicated().any():
            raise ValueError("Null or duplicate return dates")
        if frame.select((pl.col("eom") != pl.col("eom").dt.month_end()).any()).item():
            raise ValueError("Return dates must be month-end")
        if not frame.schema[column].is_numeric():
            raise ValueError("Returns must be numeric decimal values")
        if frame[column].null_count() or not frame[column].is_finite().all():
            raise ValueError("Null or non-finite returns")
    shifted = replicated.with_columns(pl.col("eom").dt.offset_by("1mo").dt.month_end())
    joined = shifted.join(benchmark, on="eom", how="inner", validate="1:1").sort("eom")
    if joined.height < 12:
        raise ValueError("At least 12 overlapping monthly observations are required")
    if any(joined[col].n_unique() < 2 for col in (factor, "benchmark_ret")):
        raise ValueError("Correlation is undefined for constant returns")
    return joined


def comparison_metrics(df: pl.DataFrame, factor: str) -> dict[str, object]:
    """Report scale-sensitive errors alongside correlation, without certifying replication."""
    x = df[factor].to_numpy()
    y = df["benchmark_ret"].to_numpy()
    error = x - y
    correlation = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("Non-finite correlation")
    return {
        "n_months": df.height,
        "first_return_month": str(df["eom"].min()),
        "last_return_month": str(df["eom"].max()),
        "return_units": "decimal",
        "correlation": correlation,
        "mean_error": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_absolute_error": float(np.max(np.abs(error))),
        "annualized_tracking_error": float(error.std(ddof=1) * np.sqrt(12)),
    }


def validate_factor(
    replicated_returns: pl.DataFrame,
    benchmark_path: Path,
    benchmark_factor_name: str,
    plot_subdir: str,
    correlation_threshold: float = 0.95,
    *,
    report_path: Path | None = None,
) -> float:
    """Evaluate a diagnostic threshold; source-methodology parity requires separate review."""
    import json

    if not 0 < correlation_threshold <= 1:
        raise ValueError("Correlation threshold must be in (0, 1]")
    benchmark_df = pl.read_csv(benchmark_path, try_parse_dates=True)
    benchmark = benchmark_df.filter(pl.col("name") == benchmark_factor_name).select(
        pl.col("date").alias("eom"), pl.col("ret").alias("benchmark_ret")
    )
    factor = next(col for col in replicated_returns.columns if col != "eom")
    joined = align_monthly_returns(replicated_returns, benchmark)
    metrics = comparison_metrics(joined, factor)
    correlation = float(metrics["correlation"])
    metrics.update(
        {
            "factor": benchmark_factor_name,
            "replicated_months": replicated_returns.height,
            "benchmark_months": benchmark.height,
            "unmatched_replicated_months": replicated_returns.height - joined.height,
            "unmatched_benchmark_months": benchmark.height - joined.height,
            "correlation_threshold": correlation_threshold,
            "correlation_threshold_passed": correlation > correlation_threshold,
            "methodology_parity": "not_certified",
        }
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))
    if report_path is not None:
        with report_path.open("x") as stream:
            json.dump(metrics, stream, indent=2, allow_nan=False)
            stream.write("\n")
    plot_cumulative_returns(joined, factor, plot_subdir)
    return correlation
