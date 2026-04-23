import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import SelfPruningNet
from evaluate import generate_report, plot_gate_distribution

def fake_cifar_loader(n=64, batch_size=32):
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size)

def test_one_step_decreases_loss():
    loader = fake_cifar_loader()
    model = SelfPruningNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    opt.zero_grad()
    loss1 = crit(model(images), labels) + 0.01 * model.sparsity_loss()
    loss1.backward()
    opt.step()
    opt.zero_grad()
    loss2 = crit(model(images), labels) + 0.01 * model.sparsity_loss()
    assert loss2 < loss1

def test_higher_lambda_larger_sparsity_contribution():
    model = SelfPruningNet()
    spar = model.sparsity_loss()
    loss_low = 0.001 * spar
    loss_high = 0.1 * spar
    assert loss_high > loss_low

def test_sparsity_grows_with_high_lambda():
    loader      = fake_cifar_loader(n=128, batch_size=32)
    model       = SelfPruningNet()
    opt         = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit        = nn.CrossEntropyLoss()
    lam         = 10.0   
    mean_gate_before = model.gate_values_flat().mean().item()
    for images, labels in loader:
        opt.zero_grad()
        loss = crit(model(images), labels) + lam * model.sparsity_loss()
        loss.backward()
        opt.step()
    mean_gate_after = model.gate_values_flat().mean().item()
    assert mean_gate_after < mean_gate_before

def test_no_nans_during_training():
    loader = fake_cifar_loader()
    model = SelfPruningNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for images, labels in loader:
        opt.zero_grad()
        loss = crit(model(images), labels) + 0.01 * model.sparsity_loss()
        loss.backward()
        opt.step()
        assert not torch.isnan(loss)

def test_checkpoint_roundtrip(tmp_path):
    model = SelfPruningNet()
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    model2 = SelfPruningNet()
    ckpt = torch.load(path)
    model2.load_state_dict(ckpt["state_dict"])
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)

def test_generate_report_creates_file(tmp_path):
    report_path = str(tmp_path / "REPORT.md")
    results = [{"lambda": 0.001, "test_accuracy": 50.0, "sparsity": 10.0}]
    generate_report(results, report_path)
    assert os.path.exists(report_path)

def test_plot_creates_png(tmp_path):
    plot_path = str(tmp_path / "plot.png")
    gates = torch.rand(100)
    plot_gate_distribution(gates, lam=0.01, save_path=plot_path)
    assert os.path.exists(plot_path)
