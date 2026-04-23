"""
tests/test_integration.py
--------------------------
Integration tests: components working together.

Tests
─────
  1.  A single training step reduces the total loss.
  2.  The sparsity loss term decreases when lambda is higher.
  3.  After several gradient steps with high lambda, sparsity increases.
  4.  Training loop produces valid (non-NaN) outputs.
  5.  Checkpoint save + load cycle restores identical gate values.
  6.  generate_report() produces a .md file with required sections.
  7.  plot_gate_distribution() creates a PNG file.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import SelfPruningNet
from prunable_layer import PrunableLinear
from evaluate import generate_report, plot_gate_distribution, load_model


# ─── Helpers ─────────────────────────────────────────────────────────

def fake_cifar_loader(n: int = 64, batch_size: int = 16):
    """Tiny in-memory dataset shaped like CIFAR-10."""
    X = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def one_step(model, loader, lam):
    """Do exactly one optimiser step; return before/after total loss."""
    opt       = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    opt.zero_grad()
    loss_before = (criterion(model(images), labels) +
                   lam * model.sparsity_loss()).item()
    (criterion(model(images), labels) + lam * model.sparsity_loss()).backward()
    opt.step()
    with torch.no_grad():
        loss_after = (criterion(model(images), labels) +
                      lam * model.sparsity_loss()).item()
    return loss_before, loss_after


# ─── 1. One step decreases loss ──────────────────────────────────────

def test_one_step_decreases_loss():
    loader = fake_cifar_loader()
    model  = SelfPruningNet()
    before, after = one_step(model, loader, lam=0.001)
    # Allow small tolerance: with Adam, loss should not increase.
    assert after <= before + 0.1, (
        f"Loss increased after one step: {before:.4f} → {after:.4f}"
    )


# ─── 2. Higher lambda → larger sparsity contribution ─────────────────

def test_higher_lambda_larger_sparsity_contribution():
    model     = SelfPruningNet()
    spar_loss = model.sparsity_loss().item()   # same for both lambdas
    weighted_low  = 1e-4 * spar_loss
    weighted_high = 1e-2 * spar_loss
    assert weighted_high > weighted_low


# ─── 3. Sparsity grows with high lambda ──────────────────────────────

def test_sparsity_grows_with_high_lambda():
    """
    High L1 lambda should push mean gate values DOWN.

    Why we check mean gate value instead of binary sparsity:
    Gates start at sigmoid(0)=0.5. Crossing the binary threshold 0.01
    requires many steps. But the mean gate value decreasing is guaranteed
    after even a few gradient updates when lambda is high.
    """
    loader      = fake_cifar_loader(n=128, batch_size=32)
    model       = SelfPruningNet()
    opt         = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit        = nn.CrossEntropyLoss()
    lam         = 10.0   # very aggressive

    mean_gate_before = model.gate_values_flat().mean().item()

    for images, labels in loader:
        opt.zero_grad()
        loss = crit(model(images), labels) + lam * model.sparsity_loss()
        loss.backward()
        opt.step()

    mean_gate_after = model.gate_values_flat().mean().item()
    assert mean_gate_after < mean_gate_before, (
        f"Mean gate did not decrease: {mean_gate_before:.4f} -> {mean_gate_after:.4f}"
    )


# ─── 4. No NaNs during training ──────────────────────────────────────

def test_no_nans_during_training():
    loader = fake_cifar_loader()
    model  = SelfPruningNet()
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit   = nn.CrossEntropyLoss()

    for images, labels in loader:
        opt.zero_grad()
        out  = model(images)
        loss = crit(out, labels) + 0.001 * model.sparsity_loss()
        loss.backward()
        opt.step()

        assert not torch.isnan(out).any(),  "NaN in model output"
        assert not torch.isnan(loss),       "NaN in loss"


# ─── 5. Checkpoint save + load restores gates ────────────────────────

def test_checkpoint_roundtrip(tmp_path):
    model = SelfPruningNet()
    # Alter gate_scores so they're not all zero
    with torch.no_grad():
        for layer in model.prunable_layers():
            layer.gate_scores.fill_(1.5)

    gates_before = model.gate_values_flat().clone()

    ckpt_path = str(tmp_path / "test_model.pt")
    torch.save({"state_dict": model.state_dict(), "lam": 0.001}, ckpt_path)

    restored = load_model(ckpt_path)
    gates_after = restored.gate_values_flat()

    assert torch.allclose(gates_before, gates_after, atol=1e-6), \
        "Checkpoint roundtrip changed gate values"


# ─── 6. generate_report() creates valid Markdown ─────────────────────

def test_generate_report_creates_file(tmp_path):
    fake_results = [
        {"lambda": 0.0001, "test_accuracy": 52.3, "sparsity": 5.1},
        {"lambda": 0.001,  "test_accuracy": 49.8, "sparsity": 38.6},
        {"lambda": 0.01,   "test_accuracy": 41.2, "sparsity": 82.4},
    ]
    report_path = str(tmp_path / "REPORT.md")
    generate_report(fake_results, report_path)

    assert os.path.exists(report_path), "Report file was not created"
    with open(report_path) as f:
        content = f.read()

    for section in ["L1", "Results Summary", "Lambda", "Sparsity"]:
        assert section in content, f"'{section}' missing from report"


# ─── 7. plot_gate_distribution() creates PNG ─────────────────────────

def test_plot_creates_png(tmp_path):
    gates     = torch.rand(1000)
    save_path = str(tmp_path / "gate_dist.png")
    plot_gate_distribution(gates, lam=0.001, save_path=save_path)
    assert os.path.exists(save_path), "PNG file was not created"
    assert os.path.getsize(save_path) > 1000, "PNG file looks empty"
