# src/runner.py
#
# Core workflow orchestration functions for the project.
#

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import polars as pl

# Import project modules and central settings
from config import settings
from src.data_validation import validate_raw_data
from src.portfolio_construction import construct_factor_portfolio
from src.validation import validate_factor


def get_data_filepath() -> Path:
    """
    Constructs the filepath for the raw characteristic data parquet file
    based on the start date in the settings.
    """
    start_fname = datetime.datetime.strptime(settings.START_DATE_STR, "%Y-%m-%d").strftime("%Y%m")
    end_fname = datetime.date.today().strftime("%Y%m")
    char_data_filename = f"jkp_char_usa_{start_fname}_{end_fname}.parquet"
    return settings.PROJECT_ROOT / "data" / "raw" / "char" / char_data_filename


def run_data_ingestion():
    """Runs the full data ingestion and validation pipeline."""
    from src.data_ingestion import download_jkp_char_data

    char_data_file = get_data_filepath()

    characteristics_to_download = list(set(char for char, _, _ in settings.FACTORS_TO_REPLICATE))

    download_jkp_char_data(
        characteristics=characteristics_to_download,
        output_path=char_data_file,
        start_date=settings.START_DATE_STR,
    )

    raw_df = pl.read_parquet(char_data_file)
    validate_raw_data(raw_df)


def print_summary_table(results: list):
    """
    Prints a final summary table of all replication correlations.

    Args:
        results (list): A list of dictionaries, each containing the results
                        of a single validation run.
    """
    print("\n" + "=" * 50)
    print(" " * 12 + "Final Replication Summary")
    print("=" * 50)

    # Create a polars DataFrame from the list of result dictionaries.
    results_df = pl.DataFrame(results)

    # Pivot the DataFrame to get the desired table structure.
    summary_table = results_df.pivot(index="Factor", on="Scheme", values="Correlation")

    # Ensure columns are in a consistent order.
    column_order = ["Factor", "ew", "vw", "vw_cap"]
    existing_columns = [col for col in column_order if col in summary_table.columns]

    print(summary_table.select(existing_columns))
    print("=" * 50)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_replication_workflow(input_file: Path, output_dir: Path) -> None:
    """Run against explicit inputs in a fresh directory and retain failed-run provenance."""
    input_file = input_file.resolve(strict=True)
    benchmarks = {scheme: path.resolve(strict=True) for scheme, path in settings.SCHEMES.items()}
    input_hashes = {str(path): _sha256(path) for path in [input_file, *benchmarks.values()]}
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=settings.PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=settings.PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    receipt = {
        "status": "running",
        "git_commit": git,
        "working_tree_dirty": bool(dirty),
        "input_sha256": input_hashes,
        "lock_sha256": _sha256(settings.PROJECT_ROOT / "uv.lock"),
        "source_sha256": {
            str(path.relative_to(settings.PROJECT_ROOT)): _sha256(path)
            for directory in ("src", "config")
            for path in sorted((settings.PROJECT_ROOT / directory).glob("*.py"))
            if path.name != "wrds_config.py"
        },
        "factors": settings.FACTORS_TO_REPLICATE,
        "correlation_threshold": settings.CORRELATION_THRESHOLD,
        "return_units": "decimal",
        "timing": "formation month t to return month-end t+1",
        "methodology_parity": "not_certified",
    }
    receipt_path = output_dir / "run.json"

    def save_receipt():
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

    save_receipt()
    all_results = []
    try:
        for scheme, benchmark_path in benchmarks.items():
            for characteristic, factor_name, direction in settings.FACTORS_TO_REPLICATE:
                name = f"{factor_name}_{scheme.upper()}"
                returns = construct_factor_portfolio(
                    input_file, characteristic, name, scheme, direction
                )
                returns.write_parquet(output_dir / f"{name.lower()}_replicated.parquet")
                correlation = validate_factor(
                    returns,
                    benchmark_path,
                    factor_name,
                    str(output_dir / "plots"),
                    settings.CORRELATION_THRESHOLD,
                    report_path=output_dir / f"{name.lower()}_validation.json",
                )
                all_results.append(
                    {"Factor": factor_name, "Scheme": scheme, "Correlation": correlation}
                )
        if any(_sha256(Path(path)) != digest for path, digest in input_hashes.items()):
            raise ValueError("Inputs changed during execution")
        receipt["status"] = "completed"
        receipt["all_correlation_thresholds_passed"] = all(
            row["Correlation"] > settings.CORRELATION_THRESHOLD for row in all_results
        )
        receipt["output_sha256"] = {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path != receipt_path
        }
        print_summary_table(all_results)
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_receipt()
