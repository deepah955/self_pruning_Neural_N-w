import torch
import torch.nn as nn
from model import SelfPruningNet
from prunable_layer import PrunableLinear

def test_prunable_output_shape():
    layer = PrunableLinear(10, 5)
    x = torch.randn(2, 10)
    out = layer(x)
    assert out.shape == (2, 5)

def test_gates_in_0_1():
    layer = PrunableLinear(10, 5)
    gates = layer.get_gates()
    assert (gates >= 0).all() and (gates <= 1).all()

def test_gradients_flow():
    layer = PrunableLinear(10, 5)
    x = torch.randn(2, 10)
    out = layer(x).sum()
    out.backward()
    assert layer.weight.grad is not None
    assert layer.gate_scores.grad is not None

def test_default_gate_value():
    layer = PrunableLinear(10, 5)
    gates = layer.get_gates()
    assert torch.allclose(gates, torch.tensor(0.5), atol=1e-3)

def test_sparsity_range():
    layer = PrunableLinear(10, 5)
    s = layer.sparsity()
    assert 0.0 <= s <= 1.0

def test_net_output_shape():
    net = SelfPruningNet()
    x = torch.randn(2, 3, 32, 32)
    out = net(x)
    assert out.shape == (2, 10)

def test_sparsity_loss_positive():
    net = SelfPruningNet()
    loss = net.sparsity_loss()
    assert loss >= 0

def test_overall_sparsity_range():
    net = SelfPruningNet()
    s = net.overall_sparsity()
    assert 0.0 <= s <= 1.0

def test_gate_values_flat_length():
    net = SelfPruningNet()
    flat = net.gate_values_flat()
    expected = sum(l.gate_scores.numel() for l in net.prunable_layers())
    assert flat.shape == (expected,)

def test_manual_gate_forced_to_zero():
    layer = PrunableLinear(10, 5)
    layer.gate_scores.data.fill_(-100.0)
    assert layer.sparsity(threshold=0.01) == 1.0

def test_bias_present():
    layer = PrunableLinear(10, 5, bias=True)
    assert layer.bias is not None

def test_bias_absent():
    layer = PrunableLinear(10, 5, bias=False)
    assert layer.bias is None

def test_no_bias_forward():
    layer = PrunableLinear(10, 5, bias=False)
    x = torch.randn(2, 10)
    out = layer(x)
    assert out.shape == (2, 5)
