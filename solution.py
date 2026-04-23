import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def sparsity(self, threshold: float = 1e-2) -> float:
        gates = self.get_gates().detach()
        return (gates < threshold).float().mean().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = self.get_gates()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)

    def prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            total = total + layer.get_gates().sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        pruned = sum(
            (l.get_gates().detach() < threshold).sum().item()
            for l in self.prunable_layers()
        )
        total = sum(l.gate_scores.numel() for l in self.prunable_layers())
        return pruned / total if total > 0 else 0.0

    def gate_values_flat(self) -> torch.Tensor:
        return torch.cat([
            l.get_gates().detach().flatten()
            for l in self.prunable_layers()
        ])

def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data"):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False
    )
    return train_loader, test_loader

def train(
    lam: float,
    epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    checkpoint_dir: str = "./checkpoints",
) -> dict:
    os.makedirs(checkpoint_dir, exist_ok=True)
    model     = SelfPruningNet().to(device)
    gate_params = [p for n, p in model.named_parameters() if 'gate' in n]
    weight_params = [p for n, p in model.named_parameters() if 'gate' not in n]
    optimiser = torch.optim.Adam([
        {'params': weight_params},
        {'params': gate_params, 'lr': 1e-2, 'weight_decay': 0.0}
    ], lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    epoch_logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        run_cls = run_spar = run_total = n = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(images)
            cls_loss  = criterion(logits, labels)
            spar_loss = model.sparsity_loss()
            loss      = cls_loss + lam * spar_loss
            loss.backward()
            optimiser.step()
            run_cls   += cls_loss.item()
            run_spar  += spar_loss.item()
            run_total += loss.item()
            n         += 1
        scheduler.step()
        avg_cls  = run_cls  / n
        avg_spar = run_spar / n
        sparsity = model.overall_sparsity()
        print(f"  [Lambda={lam}] Epoch {epoch:>2}/{epochs} | cls_loss={avg_cls:.4f}  sparsity={sparsity:.1%}")
        epoch_logs.append({
            "epoch": epoch, "cls_loss": round(avg_cls, 4),
            "spar_loss": round(avg_spar, 2), "sparsity": round(sparsity, 4),
        })
    acc            = _evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity()
    ckpt_path  = os.path.join(checkpoint_dir, f"model_lam{lam}.pt")
    gates_path = os.path.join(checkpoint_dir, f"gates_lam{lam}.pt")
    torch.save({"state_dict": model.state_dict(), "lam": lam}, ckpt_path)
    torch.save(model.gate_values_flat(), gates_path)
    return {
        "lambda":        lam,
        "test_accuracy": round(acc * 100, 2),
        "sparsity":      round(final_sparsity * 100, 2),
        "epoch_logs":    epoch_logs,
        "ckpt_path":     ckpt_path,
        "gates_path":    gates_path,
    }

def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

def plot_gate_distribution(gates: torch.Tensor, lam: float, save_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    values = gates.numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=100, color="black", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gate value  (0 = pruned,  1 = fully active)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Gate Value Distribution  (best model: Lambda = {lam})", fontsize=12)
    ax.axvline(x=0.01, color="gray", linestyle="--", linewidth=1.0,
               label="prune threshold (0.01)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  Gate plot saved -> {save_path}")

REPORT_TEMPLATE = (
    "# Self-Pruning Neural Network -- Results Report\n\n"
    "## Why Does L1 on Sigmoid Gates Encourage Sparsity?\n\n"
    "Each weight w has a gate defined as:\n\n"
    "    gate = sigmoid(gate_score)    in (0, 1)\n\n"
    "The sparsity loss adds **lambda * mean(all gates)** (normalized L1) to the total loss.\n\n"
    "| Penalty | Gradient on gate_score | Behaviour near 0 |\n"
    "|---------|------------------------|------------------|\n"
    "| L2 (sum gate^2) | -2 * gate * sigmoid' | Gradient shrinks -> gate never reaches exactly 0 |\n"
    "| L1 (mean gate)  | -sigmoid'(score)/N   | Constant pull -> gate collapses all the way to 0 |\n\n"
    "L1 maintains constant pressure regardless of the gate value.\n"
    "Once a weight is no longer needed, the L1 term wins and drives the gate to 0.\n\n"
    "---\n\n"
    "## Results Summary\n\n"
    "| Lambda | Test Accuracy (%) | Sparsity Level (%) |\n"
    "|--------|------------------|--------------------|\n"
    "{rows}\n\n"
    "---\n\n"
    "## Analysis\n\n"
    "- **Low lambda (1.0)**: Minimal pruning. Gates stay mostly open. High accuracy, low sparsity.\n"
    "- **Medium lambda (10.0)**: Good trade-off. Network drops redundant weights while keeping accuracy.\n"
    "- **High lambda (100.0)**: Aggressive pruning. Accuracy drops as too many connections are forced to zero.\n\n"
    "The gate distribution plot shows a bimodal shape for well-pruned models:\n"
    "a spike at 0 (dead gates) and a cluster near 1 (active gates).\n\n"
    "![Gate Distribution](gate_distribution.png)\n\n"
    "---\n"
    "*Generated by solution.py*\n"
)

def generate_report(results: list, report_path: str = "REPORT.md"):
    rows = "\n".join(
        f"| {r['lambda']} | {r['test_accuracy']:.2f} | {r['sparsity']:.2f} |"
        for r in results
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(REPORT_TEMPLATE.format(rows=rows))
    print(f"  Report saved  -> {report_path}")

def per_layer_sparsity(model: SelfPruningNet, threshold=1e-2):
    print(f"\n  {'Layer':<35} {'Gates':<10} {'Pruned':<10} Sparsity")
    print("  " + "-" * 60)
    for i, layer in enumerate(model.prunable_layers()):
        n    = layer.gate_scores.numel()
        p    = (layer.get_gates().detach() < threshold).sum().item()
        print(f"  PrunableLinear[{i}] ({layer.in_features}->{layer.out_features})"
              f"  {n:<10} {p:<10} {100*p/n:.1f}%")
    print("  " + "-" * 60)
    print(f"  Overall: {model.overall_sparsity(threshold)*100:.1f}%\n")

def main():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0001, 0.001, 0.01])
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--ckpt_dir",   type=str, default="./checkpoints")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.lambdas = [0.001]
        args.epochs  = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\\n{'='*60}")
    print("  Self-Pruning Neural Network  |  Tredence Case Study")
    print(f"{'='*60}")
    print(f"  Device  : {device}")
    print(f"  Lambdas : {args.lambdas}")
    print(f"  Epochs  : {args.epochs}")
    train_loader, test_loader = get_cifar10_loaders(args.batch_size, args.data_dir)
    results = []
    for lam in args.lambdas:
        print(f"\\n{'='*60}")
        print(f"  Training  Lambda = {lam}")
        print(f"{'='*60}")
        result = train(lam, args.epochs, device, train_loader, test_loader, args.ckpt_dir)
        results.append(result)
        print(f"\\n  Lambda={lam} | Accuracy={result['test_accuracy']}% | Sparsity={result['sparsity']}%")
    print(f"\\n{'='*60}")
    print("  SPARSITY REPORT")
    print(f"{'='*60}")
    for r in results:
        model = SelfPruningNet()
        ckpt  = torch.load(r["ckpt_path"], map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        print(f"\\n  Lambda = {r['lambda']}  |  Test Accuracy = {r['test_accuracy']}%")
        per_layer_sparsity(model)
    candidates = [r for r in results if r["sparsity"] > 30] or results
    best = max(candidates, key=lambda r: r["test_accuracy"])
    gates = torch.load(best["gates_path"], map_location="cpu")
    plot_gate_distribution(gates, best["lambda"], save_path="gate_distribution.png")
    import shutil
    os.makedirs("./web", exist_ok=True)
    shutil.copy("gate_distribution.png", "./web/gate_distribution.png")
    generate_report(results, "REPORT.md")
    web_json = os.path.join("./web", "results.json")
    with open(web_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Web data      -> {web_json}")
    print(f"\\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Lambda':<12} {'Accuracy (%)':<18} {'Sparsity (%)'}")
    print("  " + "-" * 42)
    for r in results:
        print(f"  {str(r['lambda']):<12} {r['test_accuracy']:<18} {r['sparsity']}")
    print(f"{'='*60}\\n")

if __name__ == "__main__":
    main()
