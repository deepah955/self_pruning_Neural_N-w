"""
main.py
-------
Single entry-point for the entire Self-Pruning Neural Network case study.

Usage
─────
    python main.py                          # use defaults
    python main.py --lambdas 0.0001 0.001 0.01 --epochs 15
    python main.py --epochs 5 --quick      # fast smoke-test run

What it does
────────────
  1. Train SelfPruningNet on CIFAR-10 for each lambda value.
  2. Report sparsity per layer.
  3. Save gate distribution plot (best model).
  4. Generate REPORT.md.
  5. Copy results.json so the web dashboard can read them.
"""

import os
import json
import argparse
import shutil
import torch

from train    import run_experiment
from evaluate import (
    load_model,
    report_sparsity,
    plot_gate_distribution,
    generate_report,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def pick_best(results: list) -> dict:
    """
    Pick the 'best' model: highest accuracy among those with >30% sparsity.
    Falls back to highest accuracy overall if no run achieves >30% sparsity.
    """
    filtered = [r for r in results if r["sparsity"] > 30]
    pool = filtered if filtered else results
    return max(pool, key=lambda r: r["test_accuracy"])


def copy_results_for_web(results: list, web_dir: str = "./web") -> None:
    """Write results.json into the web dashboard directory."""
    os.makedirs(web_dir, exist_ok=True)
    path = os.path.join(web_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Web results -> {path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network — full pipeline"
    )
    parser.add_argument(
        "--lambdas", nargs="+", type=float,
        default=[0.0001, 0.001, 0.01],
        help="Sparsity regularisation weights (space-separated floats)"
    )
    parser.add_argument("--epochs",     type=int, default=15,
                        help="Training epochs per lambda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--ckpt_dir",   type=str, default="./checkpoints")
    parser.add_argument("--web_dir",    type=str, default="./web")
    parser.add_argument("--report",     type=str, default="./REPORT.md")
    parser.add_argument("--quick",      action="store_true",
                        help="Smoke-test: 1 epoch, small lambda set")
    args = parser.parse_args()

    if args.quick:
        args.lambdas = [0.001]
        args.epochs  = 1

    # ── Step 1: Train ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SELF-PRUNING NEURAL NETWORK  |  Tredence Case Study")
    print("="*60)

    results = run_experiment(
        lambdas=args.lambdas,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        checkpoint_dir=args.ckpt_dir,
    )

    # ── Step 2: Per-layer sparsity report ─────────────────────────────
    print("\n" + "-"*60)
    print("  SPARSITY REPORT")
    print("-"*60)
    for r in results:
        ckpt = r["ckpt_path"]
        model = load_model(ckpt)
        print(f"\nLambda = {r['lambda']}  |  Accuracy = {r['test_accuracy']}%")
        report_sparsity(model)

    # ── Step 3: Gate distribution plot (best model) ───────────────────
    best = pick_best(results)
    gates = torch.load(best["gates_path"], map_location="cpu")
    plot_path = os.path.join(args.ckpt_dir, "gate_distribution.png")
    plot_gate_distribution(gates, lam=best["lambda"], save_path=plot_path)
    print(f"\n  Gate plot saved -> {plot_path}  (best model: Lambda={best['lambda']})")

    # Copy plot to web dir so the dashboard can display it
    web_img_path = os.path.join(args.web_dir, "gate_distribution.png")
    os.makedirs(args.web_dir, exist_ok=True)
    shutil.copy(plot_path, web_img_path)

    # ── Step 4: Generate Markdown report ──────────────────────────────
    report_path = generate_report(results, args.report)
    print(f"  Report saved -> {report_path}")

    # ── Step 5: Copy results for web dashboard ────────────────────────
    # Include epoch_logs for detailed chart in dashboard
    copy_results_for_web(results, args.web_dir)

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"  {'Lambda':<12} {'Test Acc (%)':<18} {'Sparsity (%)'}")
    print("  " + "-"*44)
    for r in results:
        print(f"  {str(r['lambda']):<12} {r['test_accuracy']:<18} {r['sparsity']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
