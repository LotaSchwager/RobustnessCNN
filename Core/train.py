from __future__ import print_function

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Resultado por época
# =============================================================================

@dataclass
class TrainStats:
    loss:         float
    natural_loss: float = 0.0
    robust_loss:  float = 0.0
    lambda_min:   float = float("nan")
    lambda_max:   float = float("nan")
    lambda_mean:  float = float("nan")
    extra:        dict  = field(default_factory=dict)


# =============================================================================
# Entrenamiento de una época  (agnóstico al método)
# =============================================================================

def train_one_epoch(
    model,
    device:         torch.device,
    train_loader,
    optimizer:      torch.optim.Optimizer,
    compute_loss_fn: Callable,          # función del método activo
    cfg,                                # Config (solo lectura)
    method_state,                       # estado mutable del método (o None)
    epoch:          int,
    log_interval:   int = 100,
) -> TrainStats:
    """
    Entrena una época completa usando la función de pérdida del método activo.

    compute_loss_fn debe tener la firma:
        fn(model, x, y, cfg, method_state) -> (loss_tensor, info_dict)

    info_dict puede contener cualquier subconjunto de:
        lambda_min, lambda_max, lambda_mean, loss_natural, loss_robust

    Los campos ausentes se reemplazan por 0.0 / NaN en TrainStats sin fallar.
    """
    model.train()

    seen           = 0
    epoch_loss     = 0.0
    lambda_mins:   list[float] = []
    lambda_maxs:   list[float] = []
    lambda_means:  list[float] = []
    alpha_per_class_accum: list | None = None
    beta_per_class_accum:  list | None = None

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)

        loss, info = compute_loss_fn(model, data, target, cfg, method_state)

        loss.backward()
        optimizer.step()

        bs          = data.size(0)
        epoch_loss += loss.item() * bs
        seen       += bs

        if "lambda_mean" in info:
            lambda_means.append(info["lambda_mean"])
            lambda_mins.append(info["lambda_min"])
            lambda_maxs.append(info["lambda_max"])

        if "alpha_per_class" in info:
            if alpha_per_class_accum is None:
                alpha_per_class_accum = []
                beta_per_class_accum  = []
            alpha_per_class_accum.append(info["alpha_per_class"])
            beta_per_class_accum.append(info["beta_per_class"])

        if batch_idx % log_interval == 0:
            lm = f"{np.mean(lambda_means):.4f}" if lambda_means else "n/a"
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
                f"\tλ mean: {lm}"
            )

    extra: dict = {}
    if alpha_per_class_accum is not None:
        nc = len(alpha_per_class_accum[0])
        extra["alpha_per_class"] = [float(np.mean([v[c] for v in alpha_per_class_accum])) for c in range(nc)]
        extra["beta_per_class"]  = [float(np.mean([v[c] for v in beta_per_class_accum]))  for c in range(nc)]

    stats = TrainStats(
        loss         = epoch_loss / seen,
        natural_loss = 0.0,
        robust_loss  = 0.0,
        lambda_min   = float(np.min(lambda_mins))  if lambda_mins  else float("nan"),
        lambda_max   = float(np.max(lambda_maxs))  if lambda_maxs  else float("nan"),
        lambda_mean  = float(np.mean(lambda_means)) if lambda_means else float("nan"),
        extra        = extra,
    )
    return stats


# =============================================================================
# Bucle completo  (agnóstico al método)
# =============================================================================

def run_training(
    cfg,
    model:           nn.Module,
    optimizer:       torch.optim.Optimizer,
    train_loader,
    test_loader,
    evaluator_fn:    Callable[[nn.Module, torch.device, Any], Dict[str, float]],
    metrics,
    compute_loss_fn: Callable,           # función de pérdida del método activo
    method_state     = None,             # estado mutable del método (o None)
    save_state_fn    = None,             # serializa method_state junto al ckpt
    scheduler:       Optional[Any] = None,
    start_epoch:     int = 1,
) -> None:
    """
    Bucle de entrenamiento genérico.

    No importa ni referencia ningún módulo de Metodo/ directamente.
    Toda la lógica específica del método llega a través de compute_loss_fn,
    method_state y save_state_fn.

    Agregar un método nuevo:
      1. Crear Metodo/nuevo.py con compute_loss(), make_state(),
         save_state() y load_state() de la misma firma que dtrades.py.
      2. Registrarlo en Metodo/__init__.py.
      3. main.py lo pasa aquí — train.py no necesita cambios.
    """
    device = cfg.device

    for epoch in range(start_epoch, cfg.epochs + 1):

        # ── Entrenamiento ──────────────────────────────────────────────────────
        stats = train_one_epoch(
            model            = model,
            device           = device,
            train_loader     = train_loader,
            optimizer        = optimizer,
            compute_loss_fn  = compute_loss_fn,
            cfg              = cfg,
            method_state     = method_state,
            epoch            = epoch,
            log_interval     = cfg.log_interval,
        )

        # ── Evaluación ─────────────────────────────────────────────────────────
        eval_stats = evaluator_fn(model, device, test_loader)

        # ── Métricas ───────────────────────────────────────────────────────────
        # alpha/beta por clase: específicos de D-TRADES, None para otros métodos
        metrics.update(
            epoch               = epoch,
            loss                = stats.loss,
            loss_natural        = stats.natural_loss,
            loss_robust         = stats.robust_loss,
            lambda_min          = stats.lambda_min,
            lambda_max          = stats.lambda_max,
            lambda_mean         = stats.lambda_mean,
            acc_natural         = eval_stats["natural_acc"],
            acc_robust          = eval_stats["robust_acc"],
            robust_drop         = eval_stats["robust_drop"],
            attack_success_rate = eval_stats["attack_success_rate"],
            alpha_per_class     = stats.extra.get("alpha_per_class"),
            beta_per_class      = stats.extra.get("beta_per_class"),
        )

        # ── Resumen por época ──────────────────────────────────────────────────
        lam_str = (
            f"λ=[{stats.lambda_min:.3f},{stats.lambda_mean:.3f},{stats.lambda_max:.3f}]"
            if not np.isnan(stats.lambda_mean) else ""
        )
        print(
            f"[Época {epoch:03d}/{cfg.epochs}] "
            f"Loss={stats.loss:.4f}  "
            f"AccNat={eval_stats['natural_acc']:.4f}  "
            f"AccRob={eval_stats['robust_acc']:.4f}  "
            f"{lam_str}"
        )

        # ── Checkpoint  ────────────────────────────────────────────────────────
        checkpoint_base = cfg.save_checkpoints(epoch, optimizer, model)
        if checkpoint_base is not None and save_state_fn is not None:
            save_state_fn(method_state, checkpoint_base)

        # ── Scheduler ──────────────────────────────────────────────────────────
        if scheduler is not None:
            scheduler.step()

    metrics.save_metrics()
