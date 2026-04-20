"""
Self-Pruning Neural Network on CIFAR-10
Tredence AI Engineering Internship – Case Study
 
Author: Candidate Submission
Description:
    Implements a feed-forward neural network with learnable gate parameters
    that learn to prune themselves during training via L1 sparsity regularization.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
 
# ─────────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ─────────────────────────────────────────────
 
class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate parameters.
 
    For each weight w_ij there is a corresponding gate_score g_ij.
    The effective weight used in the forward pass is:
        pruned_weight = w_ij * sigmoid(g_ij)
 
    When sigmoid(g_ij) → 0, the weight is effectively pruned.
    The optimizer updates both `weight` and `gate_scores` via backprop.
    """
 
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
 
        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
 
        # Gate scores — same shape as weight; learned independently
        # Initialized near 0 so initial gates ≈ sigmoid(0) = 0.5 (half-open)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
 
        # Standard Kaiming init for weights
        nn.init.kaiming_uniform_(self.weight, a=0.01)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gates ∈ (0, 1) — differentiable, gradient flows into gate_scores
        gates = torch.sigmoid(self.gate_scores)
 
        # Element-wise mask: prune weak weights
        pruned_weights = self.weight * gates
 
        # Standard affine transform: x @ W^T + b
        return F.linear(x, pruned_weights, self.bias)
 
    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from graph)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)
 
    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gates — encourages gates toward 0 (pruning)."""
        return torch.sigmoid(self.gate_scores).sum()
 
 
# ─────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────
 
class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32×3 = 3072 input features).
    All linear layers are replaced with PrunableLinear.
    BatchNorm and ReLU improve stability during training.
    """
 
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
 
            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
 
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
 
            PrunableLinear(256, 10),   # 10 CIFAR-10 classes
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)
 
    def get_prunable_layers(self):
        """Yield all PrunableLinear layers in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module
 
    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 gate norms across all prunable layers."""
        return sum(layer.sparsity_loss() for layer in self.get_prunable_layers())
 
    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Fraction of weights whose gate value < threshold.
        A gate below threshold is considered 'pruned'.
        """
        total, pruned = 0, 0
        for layer in self.get_prunable_layers():
            gates = layer.get_gates()
            total  += gates.numel()
            pruned += (gates < threshold).sum().item()
        return pruned / total if total > 0 else 0.0
 
    def all_gate_values(self) -> np.ndarray:
        """Collect every gate value across all layers into a flat numpy array."""
        vals = []
        for layer in self.get_prunable_layers():
            vals.append(layer.get_gates().cpu().numpy().ravel())
        return np.concatenate(vals)
 
 
# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
 
def get_cifar10_loaders(batch_size: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader
 
 
# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
 
def train_one_epoch(model, loader, optimizer, device, lam):
    model.train()
    total_loss = total_cls_loss = total_sp_loss = correct = total = 0
 
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()
 
        logits = model(images)
 
        # ── Part 2: Total Loss = CrossEntropy + λ * SparsityLoss ──
        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = model.total_sparsity_loss()
        loss     = cls_loss + lam * sp_loss
 
        loss.backward()
        optimizer.step()
 
        total_loss    += loss.item()
        total_cls_loss += cls_loss.item()
        total_sp_loss  += sp_loss.item()
 
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
 
    n = len(loader)
    return (total_loss / n,
            total_cls_loss / n,
            total_sp_loss / n,
            correct / total)
 
 
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total
 
 
# ─────────────────────────────────────────────
# Full Experiment for One Lambda
# ─────────────────────────────────────────────
 
def run_experiment(lam: float,
                   train_loader,
                   test_loader,
                   device,
                   epochs: int = 30,
                   lr: float = 1e-3) -> dict:
    print(f"\n{'='*55}")
    print(f"  Running experiment  λ = {lam}")
    print(f"{'='*55}")
 
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
 
    history = {'train_acc': [], 'test_acc': [], 'sparsity': []}
 
    for epoch in range(1, epochs + 1):
        tr_loss, cls, sp, tr_acc = train_one_epoch(model, train_loader, optimizer, device, lam)
        te_acc = evaluate(model, test_loader, device)
        spar   = model.sparsity_level()
        scheduler.step()
 
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['sparsity'].append(spar)
 
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2}/{epochs} | "
                  f"Loss {tr_loss:.4f} (cls {cls:.4f} + λ·sp {lam*sp:.4f}) | "
                  f"Train {tr_acc:.3f} | Test {te_acc:.3f} | Sparsity {spar:.1%}")
 
    final_test_acc = evaluate(model, test_loader, device)
    final_sparsity = model.sparsity_level()
    gate_vals      = model.all_gate_values()
 
    print(f"\n  ✓ Final Test Accuracy : {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    print(f"  ✓ Final Sparsity Level: {final_sparsity:.4f} ({final_sparsity*100:.2f}%)")
 
    return {
        'lambda'       : lam,
        'test_accuracy': final_test_acc,
        'sparsity'     : final_sparsity,
        'gate_vals'    : gate_vals,
        'history'      : history,
        'model'        : model,
    }
 
 
# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
 
def plot_gate_distribution(results: list, save_path: str = 'gate_distribution.png'):
    """
    Plot histogram of gate values for each λ on subplots.
    A successful result shows a spike at 0 and a cluster near 1.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
 
    colors = ['#2196F3', '#4CAF50', '#F44336']
 
    for ax, res, color in zip(axes, results, colors):
        gates = res['gate_vals']
        ax.hist(gates, bins=80, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.set_title(
            f"λ = {res['lambda']}\n"
            f"Accuracy: {res['test_accuracy']*100:.1f}%  |  "
            f"Sparsity: {res['sparsity']*100:.1f}%",
            fontsize=12
        )
        ax.set_xlabel('Gate Value', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.axvline(x=0.01, color='black', linestyle='--', alpha=0.6, label='Prune threshold (0.01)')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
 
    fig.suptitle('Distribution of Learned Gate Values\n(spike near 0 = successful pruning)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved → {save_path}")
    plt.close()
 
 
def plot_training_curves(results: list, save_path: str = 'training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2196F3', '#4CAF50', '#F44336']
    labels = [f"λ = {r['lambda']}" for r in results]
 
    for res, color, label in zip(results, colors, labels):
        h = res['history']
        axes[0].plot(h['test_acc'],  color=color, label=label)
        axes[1].plot(h['sparsity'],  color=color, label=label)
 
    axes[0].set_title('Test Accuracy over Epochs', fontsize=12)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(alpha=0.3)
 
    axes[1].set_title('Sparsity Level over Epochs', fontsize=12)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Sparsity (fraction pruned)')
    axes[1].legend(); axes[1].grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved → {save_path}")
    plt.close()
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
 
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
 
    train_loader, test_loader = get_cifar10_loaders(batch_size=256)
 
    # ── Part 3: Compare three λ values ──
    LAMBDAS = [1e-5, 1e-4, 1e-3]   # low, medium, high sparsity penalty
    EPOCHS  = 30
 
    results = []
    for lam in LAMBDAS:
        res = run_experiment(lam, train_loader, test_loader, device, epochs=EPOCHS)
        results.append(res)
 
    # ── Summary Table ──
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Lambda':<12} {'Test Accuracy':>16} {'Sparsity Level':>16}")
    print("  " + "-"*44)
    for r in results:
        print(f"  {r['lambda']:<12.5f} {r['test_accuracy']*100:>14.2f}%  {r['sparsity']*100:>14.2f}%")
    print("="*60)
 
    # ── Plots ──
    plot_gate_distribution(results, 'gate_distribution.png')
    plot_training_curves(results,   'training_curves.png')
 
    print("\nDone. Outputs: gate_distribution.png, training_curves.png")
 
 
if __name__ == '__main__':
    main()
