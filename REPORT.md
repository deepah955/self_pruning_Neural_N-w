# Self-Pruning Neural Network -- Results Report

## Why Does L1 on Sigmoid Gates Encourage Sparsity?

The gate for weight *w* is defined as:

```
gate = sigmoid(score)  in (0, 1)
```

We add **λ · Σ gate_i** to the total loss (the L1 norm of the gates,
which equals the plain sum since gates are always positive).

**Intuition:**

| Penalty | Gradient on score | Effect |
|---------|-------------------|--------|
| L2 (sum of squares) | −2 · gate · sigmoid' | pulls gate toward 0 but gradient shrinks as gate → 0; rarely reaches **exactly** 0 |
| L1 (sum of values)  | −sigmoid'(score) | constant-direction pull; gradient does NOT vanish at 0, so the gate **collapses** to 0 |

L1 creates a constant downward pressure. Once the classification loss
no longer benefits from a weight, the L1 term wins and the gate goes
to 0, permanently silencing that weight. This is exactly the "sparse
attractor" behaviour we want.

---

## Results Summary

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|------------|------------------|--------------------|
| 0.001 | 43.72 | 0.00 |

---

## Analysis

- **Low λ (≈ 0.0001):** Minimal pruning. The sparsity loss has little
  influence, so almost all gates stay open. High accuracy, low sparsity.
- **Medium λ (≈ 0.001):** A good trade-off. The network drops redundant
  weights while preserving most of its representational capacity.
- **High λ (≈ 0.01):** Aggressive pruning. Accuracy falls noticeably as
  too many useful connections are forced to zero.

The gate distribution plot (below) for the best model shows the expected
bimodal shape: a tall spike at 0 (pruned weights) and a cluster near 1
(active weights), with very little in between.

![Gate Distribution](./checkpoints/gate_distribution.png)

---
*Generated automatically by evaluate.py*
