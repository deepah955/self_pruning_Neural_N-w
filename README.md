# Self-Pruning Neural Network
*Tredence Analytics — AI Engineering Intern Case Study 2025*

---

## What this is

A PyTorch implementation of a neural network that **prunes itself during training**.
Instead of removing weights after training, the network learns *which weights to keep*
via learnable "gate" parameters that can be forced to zero by an L1 sparsity penalty.

---

## Project structure

```
self_pruning_tredence/
│
├── prunable_layer.py   # PrunableLinear — custom layer with gate mechanism
├── model.py            # SelfPruningNet — 4-layer feed-forward classifier
├── train.py            # Training loop + data loading (CIFAR-10)
├── evaluate.py         # Sparsity report, plot, and Markdown report
├── main.py             # Single entry-point for the full pipeline
│
├── tests/
│   ├── test_unit.py        # 12 unit tests (layer, gates, gradients)
│   ├── test_integration.py # 7 integration tests (train step, checkpoint, plot)
│   └── test_e2e.py         # 7 end-to-end tests (full pipeline on fake data)
│
├── web/
│   └── index.html      # Static dashboard (Netlify deploy target)
│
├── requirements.txt
├── conftest.py
├── netlify.toml
└── REPORT.md           # Auto-generated after training
```

---

## Quick start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Run all tests (no GPU, no download, ~30 seconds)

```bash
pytest tests/ -v
```

### 3 — Train the model (downloads CIFAR-10 ~170 MB on first run)

```bash
# Default: 3 lambda values, 15 epochs each
python main.py

# Custom
python main.py --lambdas 0.0001 0.001 0.01 --epochs 20

# Quick smoke-test (1 epoch, ~1 minute)
python main.py --quick
```

### 4 — View the dashboard

Open `web/index.html` in a browser.
After training, `web/results.json` will be present and the dashboard
will show real training results.

---

## Key concepts

### PrunableLinear layer

```python
gates        = sigmoid(gate_scores)      # (0, 1) per weight
pruned_weight = weight * gates
output        = pruned_weight @ x.T + bias
```

### Total loss

```
Loss = CrossEntropy(logits, labels) + λ × Σ gate_i
```

The **L1 term** (Σ gate_i) has a constant gradient pointing toward zero.
Unlike L2, it does not shrink as the gate approaches zero — it maintains
a steady push that drives the gate **all the way to 0**, completely
eliminating the corresponding weight.

---

## Results (typical run, 15 epochs)

| Lambda (λ) | Test Accuracy | Sparsity |
|------------|--------------|----------|
| 0.0001     | ~52 %        | ~5 %     |
| 0.001      | ~49 %        | ~38 %    |
| 0.01       | ~41 %        | ~82 %    |

*Results vary slightly between runs. GPU speeds up training significantly.*

---

## Deploying to Netlify

1. Push this repo to GitHub.
2. Go to [netlify.com](https://netlify.com) → **Add new site → Import from Git**.
3. Select the repo. Netlify reads `netlify.toml` and serves the `web/` directory.
4. To update results after training, commit the updated `web/results.json` and
   `web/gate_distribution.png`.

---

## Running tests

| Test file           | What it checks                                      | Time   |
|---------------------|-----------------------------------------------------|--------|
| `test_unit.py`      | Layer shape, gate range, gradients, default values  | ~3 s   |
| `test_integration.py` | Training convergence, checkpoint roundtrip, plots | ~15 s  |
| `test_e2e.py`       | Full pipeline on fake CIFAR data                    | ~20 s  |

```bash
pytest tests/ -v --tb=short
```

---

*Built for the Tredence AI Engineering Internship Case Study.*
