# Project 1: Academic Factor Replication
## Objective
The objective of this project is to replicate six configured academic investment factors using the firm-level characteristic data from the Jensen, Kelly, and Pedersen (JKP) global dataset.

The replicated factor returns are then validated against the pre-computed benchmark factors provided by the JKP authors to test the accuracy of the replication.

## Project Workflow
This project follows a professional quantitative research workflow, separating data ingestion, portfolio construction, and validation into modular scripts.

1. Data Ingestion: Download the necessary raw firm-level data (characteristics) from the Wharton Research Data Services (WRDS) database.

2. Factor Replication: Use the characteristic data to construct long-short factor portfolios based on academic methodologies. The output is a time series of returns for each replicated factor.

3. Validation: Compare the replicated factor returns against the official JKP benchmark returns. Correlation greater than 0.95 is a diagnostic threshold, not evidence of exact methodological replication. Reports also disclose overlap, decimal return units, mean error, RMSE, maximum error and annualized tracking error.

## Data Sources
This project uses two distinct datasets:

1. Firm-Level Characteristic Data
- Source: WRDS contrib.global_factor table.
- Purpose: The raw ingredients used to build our own factor portfolios.
- Key Columns: be_me (Value), ope_be (Profitability), at_gr1 (Investment), ret_12_1 (Momentum), ret_1_0 (Short-Term Reversal), and me (Market Equity for weighting).

2. JKP Benchmark Factor Data
- Source: CSV files downloaded from jkpfactors.com.
- Purpose: The "ground truth" or benchmark against which we validate our replicated factors.
- Location: Stored in the data/raw/factor/ directory.

## Reproducible development and execution

Use Python 3.12 and the pinned uv version. `uv.lock` fixes the full environment;
`requirements.txt` is only a direct-dependency compatibility list.

```sh
uvx --from uv==0.12.7 uv sync --frozen --all-groups
make UV='uvx --from uv==0.12.7 uv' check
uvx --from uv==0.12.7 uv run python main.py run-replication \
  --input-file data/raw/char/SELECTED_INPUT.parquet \
  --output-dir runs/NEW_RUN_ID
```

The input path is explicit and an existing output directory is refused. Every run
records input hashes, Git revision and dirty state, source hashes, lock hash,
factor directions, return units and output hashes. Failed runs retain a failure
receipt. Inputs are rehashed before marking completion. No credentials or licensed
data are required for CI; tests use synthetic fixtures and mocked WRDS calls.
`make format` formats active Python code; `make check` checks style and tests
without changing data or historical plots. CI runs on PRs and main pushes.

## Research contract and limits

- Observation identity is `(eom, id)`; duplicate/null keys, non-month-end dates,
  non-positive observed market equity and non-finite numeric inputs fail.
- Returns are decimal, not percent. Formation month t maps to **month-end t+1**,
  including leap years and year rollover. Benchmark joins must be one-to-one.
- At least 12 overlapping months and nonconstant series are required. This is a
  minimum diagnostic guard, not a statistically justified replication threshold.
  Missing coverage is disclosed; no interpolation or zero-filling is performed.
- The existing single-sort convention is retained: Polars **nearest** quantiles,
  NYSE 20/80 size cutoffs computed from screened data, characteristic terciles
  among non-micro stocks, at least five stocks per long/short leg, EW/VW/capped-VW,
  and source-CRSP return cutoffs at 0.1/99.9 percentiles. Invalid direction or
  unavailable winsorization references fail instead of silently changing meaning.
- Screening still depends on available lead returns, and reference cutoffs are
  computed after characteristic-specific screening. Their equivalence to the
  authors' universe, quantile convention and missing-return treatment has **not**
  been established. `src/diagnostics.py` is a legacy composition illustration
  with a different screen; it is not an oracle for the production construction.
- Reports set `methodology_parity: not_certified` even when correlation is high.
  Source-methodology comparison and a full licensed-data rerun remain necessary
  before any empirical replication claim. Existing plots are historical artifacts
  and are not regenerated or newly validated by these changes.

Primary methodology references: [JKP documentation](https://jkpfactors-data.s3.amazonaws.com/documents/Documentation.pdf)
and [authors' source repository](https://github.com/bkelly-lab/ReplicationCrisis).
The current implementation is not represented as a verified transcription of a
pinned upstream revision. CI verifies software invariants, not empirical results.
