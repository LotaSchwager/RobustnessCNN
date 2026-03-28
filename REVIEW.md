# Logic Review: D-TRADES Loss Implementation

**Branch:** `without-ema`
**Scope:** Full codebase review focusing on logical correctness and coherence of the D-TRADES training pipeline.

---

## Overall Assessment

The codebase is well-structured with a clean separation of concerns: `main.py` orchestrates, `Core/` handles training/eval/config, `Metodo/` encapsulates the loss method, and `Models/` provides architectures. The plugin-style method registration via `Metodo/__init__.py` is solid and extensible.

The D-TRADES loss formula as documented is:

```
L = L_CE(f(x), y) + (1 - f(x)_y) * [ lambda_S(x) * KL(f(x) || f(x')) + alpha * H(x) ]
```

The code in `Metodo/dtrades.py` implements this formula correctly at a structural level. However, the review found one critical gradient-flow issue and several moderate concerns detailed below.

---

## Critical: Entropy term provides zero gradient

**File:** `Metodo/dtrades.py`, lines 71-79, 256, 317

The `_normalize_batch()` function is decorated with `@torch.no_grad()`:

```python
@torch.no_grad()
def _normalize_batch(v, eps=1e-8):
    vmin = v.min()
    vmax = v.max()
    return (v - vmin) / (vmax - vmin + eps)
```

When `entropy_n = _normalize_batch(entropy)` is called at line 256, the resulting tensor has **no gradient**. Combined with the fact that `error_weight` (line 307) is also `.detach()`-ed, the term `error_weight * alpha * entropy_n` in the final loss contributes **zero gradient** to model parameters.

**What this means in practice:** The `alpha * H(x)` term in the formula has no effect on training. The model receives no gradient signal to reduce prediction uncertainty. The only gradient from the adversarial part comes from the KL divergence:

```
effective gradient source = CE(f(x), y) + (1-p_y) * (1 + beta*S_norm) * KL(f(x)||f(x'))
```

**Two possible interpretations:**

1. **If H(x) is intended as a penalty (gradient-providing term):** This is a bug. The fix would be to not use `_normalize_batch` on entropy (or create a differentiable normalization). The raw entropy already has a natural range of `[0, log(num_classes)]`, so dividing by `log(num_classes)` would normalize it to `[0, 1]` while preserving gradients.

2. **If H(x) is intended as a detached weight (like sensitivity):** Then the current behavior is correct, but the term `alpha * entropy_n` simply adds a constant offset per sample that doesn't influence optimization. It only affects the reported loss value, making it a wasted computation.

The docstring says _"Penalizacion por incertidumbre en la prediccion limpia"_ which suggests interpretation (1) -- it should provide gradient.

---

## Moderate: L2 PGD attack has a double-normalization / division-by-zero issue

**File:** `Metodo/dtrades.py`, lines 214-220

In the `l_2` attack branch:

```python
grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
safe = grad_norms.clone()
safe[safe == 0] = 1.0
delta.grad.div_(safe.view(-1, 1, 1, 1))        # first normalization
zero_mask = (grad_norms == 0).view(-1, 1, 1, 1)
delta.grad[zero_mask] = torch.randn_like(delta.grad[zero_mask])
delta.grad.div_(grad_norms.view(-1, 1, 1, 1))  # second normalization (divides by 0!)
```

Line 217 normalizes by `safe` (zeros replaced with 1.0). Line 220 normalizes again by the original `grad_norms` which **still contains zeros**. For zero-gradient samples, after the random replacement at line 219, the division at line 220 produces `inf`/`nan`.

For non-zero gradients, the effective operation is dividing by `grad_norms^2`, which is likely not the intended behavior.

**Impact:** Only affects `distance='l_2'` mode. The default is `'l_inf'`, so this path is not exercised in the current configuration.

---

## Moderate: NormalizeLayer hardcoded to 3 channels

**File:** `Models/normalize.py`, lines 7-8

```python
self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
self.register_buffer("std",  torch.tensor(std).view(1, 3, 1, 1))
```

The `.view(1, 3, 1, 1)` is hardcoded. For MNIST/FashionMNIST, `dataset_stats()` returns 1-element tuples `(0.1307,)` and `(0.3081,)`. Reshaping a 1-element tensor to `(1, 3, 1, 1)` will raise a RuntimeError.

**Impact:** CIFAR10, CIFAR100, and SVHN (3 channels) work fine. MNIST and FashionMNIST will crash at model construction. Fix: `view(1, -1, 1, 1)`.

---

## Moderate: `attack_success_rate` is actually robust error rate

**File:** `Core/eval.py`, line 44

```python
attack_success_rate = 1.0 - robust_acc  # = robust error rate
```

The standard definition of Attack Success Rate (ASR) is:

```
ASR = (correctly_classified_clean - correctly_classified_adv) / correctly_classified_clean
    = (natural_acc - robust_acc) / natural_acc
```

The current code computes `1 - robust_acc`, which is the **robust error rate**, not ASR. Example: if `natural_acc = 0.90` and `robust_acc = 0.50`, the code reports ASR = 0.50, but the true ASR = (0.90 - 0.50) / 0.90 = 0.44.

**Impact:** The metric is logged with a misleading name. Training is unaffected since this metric is only used for reporting.

---

## Minor: `gamma` parameter is dead code

**File:** `Metodo/dtrades.py`, line 163; `Core/Config.py`, line 43

The `gamma` parameter is defined in `Config`, passed to `d_trades_loss()`, received in the function signature, but **never used** inside the function body. The docstring explicitly acknowledges this: _"gamma: No se usa directamente en la perdida ... Reservado para ablaciones futuras"_.

Not a bug, but worth cleaning up or actually using (e.g., `gamma * (1 - f(x)_y)` to scale the MART-style weight).

---

## Minor: `rho` parameter in Config is unused

**File:** `Core/Config.py`, line 44

`rho` (EMA update rate for per-class stats) is defined in Config and exposed as a property, but no code anywhere in the codebase uses it. This appears to be a vestige from an earlier version that used per-class EMA statistics.

---

## Minor: Extra forward pass in batch sensitivity affects BatchNorm

**File:** `Metodo/dtrades.py`, lines 281-288

The batch sensitivity computation runs `model(x_adv_req)` as a third forward pass (after `model(x_natural)` and `model(x_adv)` on lines 234-235). Since the model is in `train()` mode at this point, BatchNorm running statistics (`running_mean`, `running_var`) are updated by this extra pass. This subtly biases the running statistics toward adversarial inputs.

**Impact:** Likely negligible but worth being aware of. Could be avoided by wrapping in `torch.no_grad()` and using `model.eval()` for just this computation, or by reusing the already-computed adversarial logits.

---

## Minor: Per-sample vs batch sensitivity mode inconsistency

**File:** `Metodo/dtrades.py`, lines 262-288

`per_sample_sensitivity=True` switches the model to `eval()` mode for the sensitivity computation (line 265), while the batch mode (`per_sample_sensitivity=False`, default) leaves it in `train()` mode. This means the sensitivity values differ not just in approximation quality but also in BatchNorm behavior.

---

## Components verified as correct

- **PGD L-inf attack** (lines 187-198): Standard implementation -- random init, KL-based gradient, sign step, projection, clamping. Correct.
- **KL direction** (lines 244-246): `F.kl_div(log_probs_adv, probs_nat)` computes `KL(nat || adv)` as documented. Correct.
- **Cross-entropy loss** (line 241): Standard CE on clean logits. Correct.
- **error_weight = (1 - p_y)** (line 307): MART-style weighting, correctly indexing `probs_nat` by true labels, correctly detached. Correct.
- **lambda_S = 1 + beta * S_norm** (line 299): Bounded to `[1, 1+beta]`, correctly detached. Correct.
- **Training loop** (`Core/train.py`): Method-agnostic, properly accumulates stats, handles absent metrics gracefully. Correct.
- **Checkpoint resume** (`main.py`, lines 122-177): Both options (auto-detect and direct path) handle epoch extraction, model/optimizer/method-state restoration, and scheduler fast-forwarding. Correct.
- **Metrics persistence** (`Core/Metrics.py`): Append-mode CSV with dynamic per-class columns. Correct for the single-run-then-resume workflow.
- **Model factory** (`Models/__init__.py`): Clean dispatch for ResNet18/50 and VGG variants. Correct.
- **Data pipeline** (`Core/Dataconfig.py`): Standard transforms without Normalize (handled by NormalizeLayer). Correct.

---

## Summary table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | Critical | `Metodo/dtrades.py` | `alpha * H(x)` has zero gradient due to `@torch.no_grad()` in `_normalize_batch` |
| 2 | Moderate | `Metodo/dtrades.py` | L2 PGD double normalization / division by zero |
| 3 | Moderate | `Models/normalize.py` | Hardcoded 3-channel view breaks MNIST/FashionMNIST |
| 4 | Moderate | `Core/eval.py` | `attack_success_rate` computes robust error rate, not ASR |
| 5 | Minor | `Metodo/dtrades.py` / `Config.py` | `gamma` parameter is dead code |
| 6 | Minor | `Core/Config.py` | `rho` parameter is unused |
| 7 | Minor | `Metodo/dtrades.py` | Extra forward pass in sensitivity updates BatchNorm stats |
| 8 | Minor | `Metodo/dtrades.py` | Eval/train mode inconsistency between sensitivity modes |
