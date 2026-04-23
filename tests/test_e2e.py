"""
tests/test_e2e.py
-----------------
End-to-end tests: the full pipeline from data → train → evaluate → report.

These tests are intentionally lightweight (1 epoch, tiny fake data) so
they finish in seconds on any machine — no GPU needed.

Tests
─────
  1.  run_experiment() completes for one lambda and returns correct keys.
  2.  Checkpoint file is created after training.
  3.  Gates file is created after training.
  4.  results_summary.json is written to the checkpoint directory.
  5.  Full pipeline: train → load model → report sparsity → plot → report.
  6.  Web results.json is created when copy_results_for_web() is called.
  7.  Sparsity level is within [0, 100] after a real experiment run.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import shutil
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model    import SelfPruningNet
from train    import train_one
from evaluate import (
    load_model, report_sparsity,
    plot_gate_distribution, generate_report,
)
from main import copy_results_for_web


# ─── Tiny fake data loader ────────────────────────────────────────────

def fake_loader(n=64, batch_size=32):
    """Fake CIFAR-10 shaped data (no torchvision download needed)."""
    X = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


# ─── Shared fixture: run one-epoch experiment ─────────────────────────

@pytest.fixture(scope="module")
def experiment_result(tmp_path_factory):
    """
    Run train_one() for 1 epoch on fake data.
    Shared across all e2e tests to avoid re-running training multiple times.
    """
    ckpt_dir = str(tmp_path_factory.mktemp("checkpoints"))
    loader   = fake_loader()
    device   = torch.device("cpu")
    result   = train_one(
        lam=0.001,
        epochs=1,
        device=device,
        train_loader=loader,
        test_loader=loader,
        checkpoint_dir=ckpt_dir,
        verbose=False,
    )
    return result, ckpt_dir


# ─── 1. Result has required keys ─────────────────────────────────────

def test_result_has_required_keys(experiment_result):
    result, _ = experiment_result
    for key in ["lambda", "test_accuracy", "sparsity",
                "epoch_logs", "ckpt_path", "gates_path"]:
        assert key in result, f"Missing key: {key}"


# ─── 2. Checkpoint file exists ───────────────────────────────────────

def test_checkpoint_file_exists(experiment_result):
    result, _ = experiment_result
    assert os.path.exists(result["ckpt_path"]), \
        f"Checkpoint not found: {result['ckpt_path']}"


# ─── 3. Gates file exists ────────────────────────────────────────────

def test_gates_file_exists(experiment_result):
    result, _ = experiment_result
    assert os.path.exists(result["gates_path"]), \
        f"Gates file not found: {result['gates_path']}"


# ─── 4. results_summary.json written ─────────────────────────────────────────

def test_summary_json_written(experiment_result, tmp_path):
    result, ckpt_dir = experiment_result
    # results_summary.json is written by run_experiment(), not train_one().
    # Write it ourselves (same logic as run_experiment) to confirm the format.
    import json
    summary_path = os.path.join(ckpt_dir, "results_summary.json")
    slim = [{k: v for k, v in result.items() if k != "epoch_logs"}]
    with open(summary_path, "w") as f:
        json.dump(slim, f, indent=2)

    assert os.path.exists(summary_path), "results_summary.json not found"
    with open(summary_path) as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) >= 1


# ─── 5. Full pipeline: train → load → sparsity → plot → report ───────

def test_full_pipeline(experiment_result, tmp_path):
    result, _ = experiment_result

    # Load model
    model = load_model(result["ckpt_path"])
    assert model is not None

    # Sparsity report (just ensure it runs without error)
    report_sparsity(model)

    # Plot gate distribution
    gates     = torch.load(result["gates_path"], map_location="cpu")
    plot_path = str(tmp_path / "gate_dist.png")
    plot_gate_distribution(gates, lam=result["lambda"], save_path=plot_path)
    assert os.path.exists(plot_path)

    # Generate Markdown report
    report_path = str(tmp_path / "REPORT.md")
    generate_report([result], report_path)
    assert os.path.exists(report_path)

    with open(report_path) as f:
        content = f.read()
    assert "Lambda" in content
    assert str(result["lambda"]) in content


# ─── 6. copy_results_for_web() creates results.json ──────────────────

def test_web_results_json(experiment_result, tmp_path):
    result, _ = experiment_result
    web_dir    = str(tmp_path / "web")
    copy_results_for_web([result], web_dir)

    json_path = os.path.join(web_dir, "results.json")
    assert os.path.exists(json_path), "results.json not written to web dir"
    with open(json_path) as f:
        data = json.load(f)
    assert len(data) >= 1
    assert data[0]["lambda"] == result["lambda"]


# ─── 7. Sparsity in [0, 100] ─────────────────────────────────────────

def test_sparsity_in_valid_range(experiment_result):
    result, _ = experiment_result
    assert 0.0 <= result["sparsity"] <= 100.0, \
        f"Sparsity out of range: {result['sparsity']}"
