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
    metrics,                            # instancia de Metrics
    log_interval:   int = 100,
) -> TrainStats:
    """
    Entrena una época completa usando la función de pérdida del método activo.

    compute_loss_fn debe tener la firma:
        fn(model, x, y, cfg, method_state) -> (loss_tensor, info_dict)

    info_dict contiene arrays numpy per-sample y estadísticas que se
    pasan directamente a metrics.record_batch().
    """
    model.train()

    seen           = 0
    epoch_loss     = 0.0
    lambda_means:  list[float] = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)

        loss, info = compute_loss_fn(model, data, target, cfg, method_state)

        loss.backward()
        optimizer.step()

        bs          = data.size(0)
        epoch_loss += loss.item() * bs
        seen       += bs

        # ── Registrar métricas por batch ───────────────────────────────────
        if "lam" in info:
            metrics.record_batch(epoch, batch_idx, info)
            lambda_means.append(float(np.mean(info["lam"])))

        if batch_idx % log_interval == 0:
            lm = f"{lam_mean:.4f}" if not np.isnan(lam_mean) else "n/a"
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
                f"\tλ mean: {lm}"
            )

    # Resumen de la época para TrainStats (retrocompatibilidad)
    lam_min  = float("nan")
    lam_max  = float("nan")
    lam_mean = float("nan")
    if lambda_means:
        lam_mean = float(np.mean(lambda_means))
        # min/max se estiman de los promedios por batch (aproximación razonable)
        lam_min  = float(np.min(lambda_means))
        lam_max  = float(np.max(lambda_means))

    stats = TrainStats(
        loss         = epoch_loss / seen,
        natural_loss = 0.0,
        robust_loss  = 0.0,
        lambda_min   = lam_min,
        lambda_max   = lam_max,
        lambda_mean  = lam_mean,
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
            metrics          = metrics,
            log_interval     = cfg.log_interval,
        )

        # ── Evaluación ─────────────────────────────────────────────────────────
        eval_stats = evaluator_fn(model, device, test_loader)

        # ── Métricas por época (promedios de batches) ──────────────────────────
        metrics.record_epoch(epoch)

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
