# inspect_benchmark.py
from pathlib import Path

import polars as pl


def main() -> None:
    # Define the path to your benchmark file
    project_root = Path(__file__).resolve().parent
    benchmark_file = project_root / "data" / "raw" / "factor" / "usa_all_factors_monthly_vw_cap.csv"

    # Load the data
    df = pl.read_csv(benchmark_file)

    # Get and print all unique factor names
    unique_names = df.get_column("name").unique().sort()

    print("Available factor names in the benchmark file:")
    for name in unique_names:
        print(f"- {name}")


if __name__ == "__main__":
    main()
