"""Network-free input integrity, SQL binding and output provenance tests."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src import data_ingestion, runner


@pytest.mark.parametrize(
    "kwargs",
    [
        {"country": "USA' OR 1=1--"},
        {"characteristics": ['x" FROM secrets--']},
        {"start_date": "2020-01-01'"},
    ],
)
def test_invalid_sql_input_fails_before_connection(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(
        data_ingestion.wrds, "Connection", lambda **kw: pytest.fail("Unexpected network")
    )
    options = {
        "characteristics": ["be_me"],
        "output_path": tmp_path / "input.parquet",
        "wrds_username": "test-user",
        **kwargs,
    }
    with pytest.raises(ValueError):
        data_ingestion.download_jkp_char_data(**options)


def test_download_binds_values_closes_connection_and_preserves_existing_input(
    tmp_path, monkeypatch
):
    calls = {}

    class FakeDB:
        def raw_sql(self, query, **kwargs):
            calls.update(query=query, **kwargs)
            return pl.DataFrame(
                {"eom": [date(2020, 1, 31)], "id": [1], "me": [2.0], "ret_exc_lead1m": [0.1]}
            ).to_pandas()

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(data_ingestion.wrds, "Connection", lambda **kw: FakeDB())
    path = tmp_path / "input.parquet"
    data_ingestion.download_jkp_char_data(["be_me"], path, "test-user", "2020-01-01")
    assert "%(country)s" in calls["query"] and "%(start_date)s" in calls["query"]
    assert calls["params"] == {"country": "USA", "start_date": "2020-01-01"}
    assert calls["closed"]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        data_ingestion.download_jkp_char_data(["be_me"], path, "test-user")
    assert path.read_bytes() == original


@pytest.fixture
def run_inputs(tmp_path, monkeypatch):
    source = tmp_path / "input.parquet"
    source.write_bytes(b"input fixture")
    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text("benchmark fixture")
    monkeypatch.setattr(runner.settings, "SCHEMES", {"ew": benchmark})
    monkeypatch.setattr(runner.settings, "FACTORS_TO_REPLICATE", [("signal", "value", "long")])
    monkeypatch.setattr(
        runner,
        "construct_factor_portfolio",
        lambda *args: pl.DataFrame({"eom": [date(2020, 1, 31)], "factor": [0.1]}),
    )
    return source, tmp_path / "run"


def test_run_records_identity_and_never_overwrites(run_inputs, monkeypatch):
    source, output = run_inputs
    monkeypatch.setattr(runner, "validate_factor", lambda *args, **kwargs: 0.99)
    runner.run_replication_workflow(source, output)
    receipt = json.loads((output / "run.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["methodology_parity"] == "not_certified"
    assert receipt["input_sha256"][str(source)] == runner._sha256(source)
    assert receipt["source_sha256"]["src/runner.py"] == runner._sha256(Path(runner.__file__))
    assert receipt["output_sha256"]
    original = (output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_replication_workflow(source, output)
    assert (output / "run.json").read_bytes() == original


def test_failed_run_retains_failure_receipt(run_inputs, monkeypatch):
    source, output = run_inputs

    def fail(*args, **kwargs):
        raise ValueError("bad benchmark")

    monkeypatch.setattr(runner, "validate_factor", fail)
    with pytest.raises(ValueError, match="bad benchmark"):
        runner.run_replication_workflow(source, output)
    receipt = json.loads((output / "run.json").read_text())
    assert receipt["status"] == "failed"
    assert "bad benchmark" in receipt["error"]


def test_mid_run_input_mutation_cannot_be_success(run_inputs, monkeypatch):
    source, output = run_inputs

    def mutate(*args, **kwargs):
        source.write_bytes(b"changed")
        return 1.0

    monkeypatch.setattr(runner, "validate_factor", mutate)
    with pytest.raises(ValueError, match="Inputs changed"):
        runner.run_replication_workflow(source, output)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
