# main.py
#
# Command-line interface for the factor replication project.
#

import argparse
from pathlib import Path

from src import runner


def main():
    """
    Parses command-line arguments and calls the appropriate
    runner function to execute a workflow.
    """
    parser = argparse.ArgumentParser(description="Factor Replication Project CLI")

    # Create subparsers to handle different commands.
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Command to run the data ingestion process.
    subparsers.add_parser("ingest-data", help="Download and validate raw data from WRDS.")

    # Command to run the full replication and validation workflow.
    replication = subparsers.add_parser(
        "run-replication", help="Run a recorded research comparison."
    )
    replication.add_argument("--input-file", type=Path, required=True)
    replication.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    # Call the correct function from the runner based on the command.
    if args.command == "ingest-data":
        runner.run_data_ingestion()
    elif args.command == "run-replication":
        runner.run_replication_workflow(args.input_file, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
