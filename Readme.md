# 🧠 Self-Pruning Neural Network — Tredence AI Engineering Internship Case Study

A PyTorch implementation of a neural network that **prunes itself during training**
using learnable gate parameters, temperature-sharpened sigmoids, and L1 sparsity
regularization — evaluated on CIFAR-10.

---

## 📁 Repository Structure
.
├── self_pruning_nn.ipynb   # Main notebook (all code + analysis + plots)
├── self_pruning_results.png  # Auto-generated results dashboard (after run)
├── data/                   # CIFAR-10 auto-downloaded here
└── README.md

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib tqdm
```

### 2. Run the notebook

Open `self_pruning_nn.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

```bash
jupyter notebook self_pruning_nn.ipynb
```

> **Runtime estimate:** ~5 min on GPU · ~15–30 min on CPU

---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `torchvision` | CIFAR-10 dataset + transforms |
| `matplotlib` | Gate distribution + results plots |
| `tqdm` | Training progress bars |

Python ≥ 3.8 recommended.

---

## 🔑 Core Concepts

### PrunableLinear Layer

A drop-in replacement for `nn.Linear` with a learnable gate for every weight:
gate  = sigmoid(gate_score × temperature)
output = F.linear(x,  weight × gate,  bias)

- `gate_scores` are registered as `nn.Parameter` → updated by the optimizer
- Gradients flow through **both** `weight` and `gate_scores`
- At **eval time**, gates are hard-thresholded to `{0, 1}` (Straight-Through Estimator)

### Sparsity Loss (L1)
Total Loss = CrossEntropyLoss  +  λ × SparsityLoss
SparsityLoss = mean of all gate values across all PrunableLinear layers

The **L1 norm** (sum of gates) encourages *exact* zeros because its gradient
is a constant `+1` for any positive gate — unlike L2, which has a vanishing
gradient and only asymptotically approaches zero.

---

## ✨ Unique Design Choices

| Feature | Description |
|---|---|
| **Temperature annealing** | Temperature T is cosine-annealed from 1 → 8. Early training uses soft gates (smooth gradients); later training sharpens them to near-binary, committing pruning decisions |
| **λ warm-up (5 epochs)** | Sparsity penalty ramps linearly from 0 → λ, preventing aggressive pruning before the classifier has learned useful features |
| **ResNet-style skip connections** | The architecture uses residual blocks built entirely from `PrunableLinear` — including the projection shortcuts |
| **Per-layer sparsity tracking** | Records which layers prune most — task-specific deeper layers stay denser; early mixing layers prune more aggressively |
| **5-panel results dashboard** | Gate histograms · loss curves · sparsity evolution · accuracy-sparsity scatter · per-layer bar chart |

---

## 🏗️ Architecture
Input: 32×32×3 image → flatten → (B, 3072)
Stem:     PrunableLinear(3072 → 512) + BN + GELU
Block 1:  PrunableResidualBlock(512 → 256)  [skip projection]
Block 2:  PrunableResidualBlock(256 → 256)  [identity skip]
Block 3:  PrunableResidualBlock(256 → 128)  [skip projection]
Head:     PrunableLinear(128 → 64) + GELU
PrunableLinear(64  → 10)
All linear layers → PrunableLinear (self-pruning)

---

## 🧪 Experiments

Three values of λ are tested to demonstrate the sparsity–accuracy trade-off:

| Run | λ | Expected behaviour |
|---|---|---|
| Low | `1e-4` | Dense network, highest accuracy |
| Medium | `5e-4` | Bimodal gate distribution, balanced trade-off |
| High | `2e-3` | Aggressively pruned, accuracy drops |

Results are printed as a table and visualized in `self_pruning_results.png`.

---

## 📊 Expected Output
────────────────────────────────────────────────────────────
Lambda        Test Accuracy      Sparsity Level (%)
────────────────────────────────────────────────────────────
1e-04              ~52–55%               ~10–20%   (low)
5e-04              ~48–52%               ~40–60%   (medium)
2e-03              ~38–45%               ~70–85%   (high)
────────────────────────────────────────────────────────────

> Exact values depend on hardware, random seed, and number of epochs.

---

## 📈 Why L1 Encourages Sparsity

The gate is always in `(0, 1)` after sigmoid, so `|gate| = gate`.
Minimizing the L1 penalty means minimizing the **sum of gate values**.

- **L2** penalty gradient → `2·gate → 0` as gate → 0 (vanishing pull, never truly zero)
- **L1** penalty gradient → `sign(gate) = +1` always (constant pull → exact zero)

This is the same principle behind LASSO regression producing sparse coefficients.
Combined with temperature annealing, gates are forced to commit to `0` or `1` by
the end of training, producing a genuinely sparse network.

---

## 🗂️ Notebook Structure

| Cell | Content |
|---|---|
| 1 | Imports, seeds, device setup |
| 2 | CIFAR-10 data loading + augmentation |
| 3 | `PrunableLinear` implementation |
| 4 | `SelfPruningNet` (ResNet-style MLP) |
| 5 | Sparsity loss · λ warm-up · temperature schedule |
| 6 | `train_one_epoch` · `evaluate` functions |
| 7 | Full training loop (3 λ runs) |
| 8 | Results summary table + per-layer breakdown |
| 9 | 5-panel visualization dashboard |
| 10 | Markdown analysis report |

---

## 📝 Submission

Built for the **Tredence AI Engineering Internship — 2025 Cohort**.

- All code is original and independently written
- No external pruning libraries used — everything implemented from scratch in PyTorch
- Gradients verified to flow through both `weight` and `gate_scores` parameters

---

## 📄 License

MIT