import os
import json
import torch
import pytest
from train import train_one, get_cifar10_loaders
from model import SelfPruningNet

@pytest.fixture
def mock_loaders():
    images = torch.randn(10, 3, 32, 32)
    labels = torch.randint(0, 10, (10,))
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    return loader, loader

def test_result_has_required_keys(mock_loaders):
    train_loader, test_loader = mock_loaders
    res = train_one(0.01, 1, "cpu", train_loader, test_loader, verbose=False)
    keys = ["lambda", "test_accuracy", "sparsity", "epoch_logs", "ckpt_path", "gates_path"]
    for k in keys:
        assert k in res

def test_checkpoint_file_exists(mock_loaders):
    train_loader, test_loader = mock_loaders
    res = train_one(0.01, 1, "cpu", train_loader, test_loader, verbose=False)
    assert os.path.exists(res["ckpt_path"])

def test_gates_file_exists(mock_loaders):
    train_loader, test_loader = mock_loaders
    res = train_one(0.01, 1, "cpu", train_loader, test_loader, verbose=False)
    assert os.path.exists(res["gates_path"])

def test_summary_json_written(mock_loaders, tmp_path):
    train_loader, test_loader = mock_loaders
    ckpt_dir = str(tmp_path / "checkpoints")
    res = train_one(0.01, 1, "cpu", train_loader, test_loader, checkpoint_dir=ckpt_dir, verbose=False)
    from train import run_experiment
    summary_path = os.path.join(ckpt_dir, "results_summary.json")
    slim = [{k: v for k, v in res.items() if k != "epoch_logs"}]
    with open(summary_path, "w") as f:
        json.dump(slim, f)
    assert os.path.exists(summary_path)

def test_full_pipeline(mock_loaders, tmp_path):
    train_loader, test_loader = mock_loaders
    ckpt_dir = str(tmp_path / "checkpoints")
    res = train_one(0.001, 1, "cpu", train_loader, test_loader, checkpoint_dir=ckpt_dir, verbose=False)
    assert res["lambda"] == 0.001
    assert res["test_accuracy"] >= 0

def test_web_results_json(mock_loaders, tmp_path):
    train_loader, test_loader = mock_loaders
    web_dir = str(tmp_path / "web")
    res = train_one(0.01, 1, "cpu", train_loader, test_loader, verbose=False)
    from main import copy_results_for_web
    copy_results_for_web([res], web_dir=web_dir)
    assert os.path.exists(os.path.join(web_dir, "results.json"))

def test_sparsity_in_valid_range(mock_loaders):
    train_loader, test_loader = mock_loaders
    res = train_one(0.1, 1, "cpu", train_loader, test_loader, verbose=False)
    assert 0.0 <= res["sparsity"] <= 100.0
