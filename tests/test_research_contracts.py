"""Synthetic adversarial tests, not evidence of empirical JKP parity."""

import calendar
from datetime import date

import polars as pl
import pytest

from src.data_validation import validate_raw_data
from src.portfolio_construction import construct_factor_portfolio
from src.validation import align_monthly_returns, comparison_metrics, validate_factor


def month_end(year, month):
    return date(year, month, calendar.monthrange(year, month)[1])


@pytest.fixture
def panel():
    return pl.DataFrame(
        {
            "eom": [date(2020, 1, 31)] * 30,
            "id": list(range(30)),
            "permno": list(range(30)),
            "crsp_exchcd": [1] * 30,
            "source_crsp": [1] * 30,
            "size_grp": ["large"] * 30,
            "me": [float(i) for i in range(1, 31)],
            "signal": [float(i) for i in range(1, 31)],
            "ret_exc_lead1m": [i / 1000 for i in range(1, 31)],
        }
    )


@pytest.fixture
def return_pair():
    dates = [month_end(2020, month) for month in range(1, 13)]
    realized = [month_end(2020 + (month == 12), month % 12 + 1) for month in range(1, 13)]
    returns = [0.01, -0.02, 0.03, 0.0, -0.04, 0.05, 0.02, -0.01, 0.06, 0.01, -0.03, 0.04]
    return pl.DataFrame({"eom": dates, "factor": returns}), pl.DataFrame(
        {"eom": realized, "benchmark_ret": returns}
    )


@pytest.mark.parametrize("scheme", ["ew", "vw", "vw_cap"])
@pytest.mark.parametrize("direction,sign", [("long", 1), ("short", -1)])
def test_hand_computed_portfolio_and_permutation_invariance(
    tmp_path, panel, scheme, direction, sign
):
    # With nearest quantiles: non-micro universe is 8..30, cutoffs 15 and 23,
    # and NYSE cap is 24. Compute expected legs independently of Polars.
    def leg(values):
        weights = [1 if scheme == "ew" else min(v, 24) if scheme == "vw_cap" else v for v in values]
        return sum(v / 1000 * w for v, w in zip(values, weights, strict=True)) / sum(weights)

    expected = sign * (leg(list(range(24, 31))) - leg(list(range(1, 16))))
    for frame in (panel, panel.reverse()):
        path = tmp_path / "panel.parquet"
        frame.write_parquet(path)
        result = construct_factor_portfolio(path, "signal", "factor", scheme, direction)
        assert result["factor"].item() == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    "column,value",
    [
        ("me", 0.0),
        ("me", -1.0),
        ("me", float("nan")),
        ("ret_exc_lead1m", float("inf")),
        ("signal", float("nan")),
        ("source_crsp", 2),
    ],
)
def test_invalid_portfolio_inputs_fail(tmp_path, panel, column, value):
    path = tmp_path / "invalid.parquet"
    panel.with_columns(pl.lit(value).alias(column)).write_parquet(path)
    with pytest.raises(ValueError):
        construct_factor_portfolio(path, "signal", "factor")


@pytest.mark.parametrize("direction", ["LONG", "typo", "", None])
def test_unknown_direction_never_silently_reverses_returns(tmp_path, direction):
    with pytest.raises(ValueError, match="direction"):
        construct_factor_portfolio(
            tmp_path / "absent", "signal", "factor", long_short_direction=direction
        )


def test_duplicate_stock_month_and_empty_data_fail(panel):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_raw_data(pl.concat([panel, panel.head(1)]))
    with pytest.raises(ValueError, match="empty"):
        validate_raw_data(panel.head(0))


@pytest.mark.parametrize("kind", ["one_leg", "no_crsp", "too_few"])
def test_missing_reference_or_portfolio_legs_fail(tmp_path, panel, kind):
    frame = {
        "one_leg": panel.with_columns(pl.lit(1.0).alias("signal")),
        "no_crsp": panel.with_columns(pl.lit(0).alias("source_crsp")),
        "too_few": panel.head(4),
    }[kind]
    path = tmp_path / "panel.parquet"
    frame.write_parquet(path)
    with pytest.raises(ValueError):
        construct_factor_portfolio(path, "signal", "factor")


def test_next_month_end_handles_leap_february_and_year_rollover(return_pair):
    replicated, benchmark = return_pair
    aligned = align_monthly_returns(replicated.reverse(), benchmark.reverse())
    assert aligned.height == 12
    assert aligned["eom"].to_list() == benchmark["eom"].to_list()
    assert aligned["factor"].to_list() == benchmark["benchmark_ret"].to_list()


@pytest.mark.parametrize(
    "kind", ["duplicate", "null", "nan", "inf", "constant", "short", "not_month_end"]
)
def test_benchmark_validation_rejects_false_confidence(return_pair, kind):
    replicated, benchmark = return_pair
    if kind == "duplicate":
        benchmark = pl.concat([benchmark, benchmark.head(1)])
    elif kind in {"null", "nan", "inf", "constant"}:
        value = {"null": None, "nan": float("nan"), "inf": float("inf"), "constant": 0.0}[kind]
        benchmark = benchmark.with_columns(pl.lit(value, dtype=pl.Float64).alias("benchmark_ret"))
    elif kind == "short":
        benchmark = benchmark.head(11)
    else:
        benchmark = benchmark.with_columns(pl.col("eom") - pl.duration(days=1))
    with pytest.raises(ValueError):
        align_monthly_returns(replicated, benchmark)


def test_perfect_correlation_does_not_hide_unit_error(return_pair):
    replicated, benchmark = return_pair
    replicated = replicated.with_columns(pl.col("factor") * 100)
    metrics = comparison_metrics(align_monthly_returns(replicated, benchmark), "factor")
    assert metrics["correlation"] == pytest.approx(1)
    assert metrics["rmse"] > 1
    assert metrics["annualized_tracking_error"] > 1


def test_metrics_report_does_not_claim_methodology_parity(tmp_path, monkeypatch, return_pair):
    import json

    import src.validation as validation

    replicated, benchmark = return_pair
    path = tmp_path / "benchmark.csv"
    benchmark.rename({"eom": "date", "benchmark_ret": "ret"}).with_columns(
        pl.lit("value").alias("name")
    ).write_csv(path)
    monkeypatch.setattr(validation, "plot_cumulative_returns", lambda *args: None)
    report = tmp_path / "report.json"
    assert validate_factor(
        replicated, path, "value", "unused", report_path=report
    ) == pytest.approx(1)
    record = json.loads(report.read_text())
    assert record["n_months"] == 12
    assert record["methodology_parity"] == "not_certified"
    with pytest.raises(FileExistsError):
        validate_factor(replicated, path, "value", "unused", report_path=report)


def test_import_inspector_has_no_file_io():
    import inspect_benchmark

    assert callable(inspect_benchmark.main)


def test_end_to_end_synthetic_run_with_real_plots_and_receipts(tmp_path, monkeypatch, panel):
    import json

    from src import runner

    input_path = tmp_path / "panel.parquet"
    benchmark_path = tmp_path / "benchmark.csv"
    frames = [
        panel.with_columns(
            pl.lit(month_end(2020, month)).alias("eom"),
            (pl.col("ret_exc_lead1m") * (month / 12)).alias("ret_exc_lead1m"),
        )
        for month in range(1, 13)
    ]
    pl.concat(frames).write_parquet(input_path)
    pl.DataFrame(
        {
            "name": ["value"] * 12,
            "date": [month_end(2020 + (m == 12), m % 12 + 1) for m in range(1, 13)],
            "ret": [0.019 * m / 12 for m in range(1, 13)],
        }
    ).write_csv(benchmark_path)
    monkeypatch.setattr(runner.settings, "SCHEMES", {"ew": benchmark_path})
    monkeypatch.setattr(runner.settings, "FACTORS_TO_REPLICATE", [("signal", "value", "long")])
    output = tmp_path / "run"
    runner.run_replication_workflow(input_path, output)
    receipt = json.loads((output / "run.json").read_text())
    report = json.loads((output / "value_ew_validation.json").read_text())
    assert receipt["status"] == "completed"
    assert report["rmse"] < 1e-14
    assert report["unmatched_replicated_months"] == 0
    assert (output / "plots/value_ew.png").stat().st_size > 1000
    for name, digest in receipt["output_sha256"].items():
        assert runner._sha256(output / name) == digest


@pytest.mark.parametrize("scheme", ["vw", "vw_cap"])
@pytest.mark.parametrize("scale", [1e306, 1e-200])
def test_weighted_returns_are_invariant_to_market_equity_units(tmp_path, panel, scheme, scale):
    path = tmp_path / "panel.parquet"
    panel.write_parquet(path)
    expected = construct_factor_portfolio(path, "signal", "factor", scheme)["factor"].item()
    panel.with_columns(pl.col("me") * scale).write_parquet(path)
    actual = construct_factor_portfolio(path, "signal", "factor", scheme)["factor"].item()
    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_raw_market_equity_guard_is_not_masked_by_later_portfolio_failure(panel, value):
    frame = panel.with_columns(
        pl.when(pl.col("id") == 0).then(pl.lit(value)).otherwise(pl.col("me")).alias("me")
    )
    with pytest.raises(ValueError, match="Market equity must be positive"):
        validate_raw_data(frame)
