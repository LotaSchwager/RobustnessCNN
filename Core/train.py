from __future__ import print_function

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from Metodo.dtrades import d_trades_loss


# =============================================================================
# Tipos de resultado por época
# =============================================================================

@dataclass
class TrainStats:
    loss:         float
    natural_loss: float
    robust_loss:  float
    lambda_min:   float
    lambda_max:   float
    lambda_mean:  float


# =============================================================================
# Entrenamiento de una época
# =============================================================================

def train_one_epoch(
    model:        nn.Module,
    device:       torch.device,
    train_loader,
    optimizer:    torch.optim.Optimizer,
    epsilon:      float,
    step_size:    float,
    num_steps:    int,
    alpha:        float,
    beta:         float,
    epoch:        int,
    log_interval: int = 100,
) -> TrainStats:
    """
    Entrena el modelo durante una época con D-TRADES.

    Parámetros
    ----------
    alpha, beta : pesos actuales de la metaheurística.
    """
    model.train()

    seen              = 0
    epoch_loss        = 0.0
    epoch_natural     = 0.0
    epoch_robust      = 0.0
    lambda_min_list:  list[float] = []
    lambda_max_list:  list[float] = []
    lambda_mean_list: list[float] = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)

        loss, lambda_value, loss_natural, loss_robust_dynamic = d_trades_loss(
            model        = model,
            x_natural    = data,
            y            = target,
            optimizer    = optimizer,
            step_size    = step_size,
            epsilon      = epsilon,
            perturb_steps= num_steps,
            alpha        = alpha,
            beta         = beta,
        )

        loss.backward()
        optimizer.step()

        bs             = data.size(0)
        epoch_loss    += loss.item() * bs
        epoch_natural += loss_natural.item() * bs
        epoch_robust  += loss_robust_dynamic.item() * bs
        seen          += bs

        lambda_min_list.append(lambda_value.min().item())
        lambda_max_list.append(lambda_value.max().item())
        lambda_mean_list.append(lambda_value.mean().item())

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
                f"\talpha: {alpha:.4f}  beta: {beta:.4f}"
                f"\tλ mean: {np.mean(lambda_mean_list):.4f}"
                f"\tλ min: {np.min(lambda_min_list):.4f}"
                f"\tλ max: {np.max(lambda_max_list):.4f}"
            )

        # Modo DEBUG: salir después de 5 batches
        if os.getenv("DEBUG", "false").lower() == "true" and batch_idx >= 4:
            break

    return TrainStats(
        loss         = epoch_loss    / seen,
        natural_loss = epoch_natural / seen,
        robust_loss  = epoch_robust  / seen,
        lambda_min   = float(np.min(lambda_min_list)),
        lambda_max   = float(np.max(lambda_max_list)),
        lambda_mean  = float(np.mean(lambda_mean_list)),
    )


# =============================================================================
# Bucle de entrenamiento completo
# =============================================================================

def run_training(
    cfg,
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    train_loader,
    test_loader,
    evaluator_fn: Callable[[nn.Module, torch.device, Any], Dict[str, float]],
    metrics,
    scheduler:    Optional[Any] = None,
    metaheuristic = None,
    start_epoch:  int = 1,
) -> None:
    """
    Entrena el modelo durante el número de épocas indicado en cfg.

    Si se proporciona una metaheurística, se llama a metaheuristic.update(loss)
    al final de cada época para actualizar cfg.alpha y cfg.beta dinámicamente.

    Parámetros
    ----------
    metaheuristic : BaseMetaheuristic | None
        Instancia de cualquier subclase de BaseMetaheuristic.
        Si es None, alpha y beta permanecen fijos durante todo el entrenamiento.
    """
    device = cfg.device

    for epoch in range(start_epoch, cfg.epochs + 1):

        # ── Usar alpha/beta actuales (pueden ser actualizados por la metaheurística) ──
        alpha = cfg.alpha
        beta  = cfg.beta

        # ── Entrenamiento ──────────────────────────────────────────────────────────
        stats = train_one_epoch(
            model        = model,
            device       = device,
            train_loader = train_loader,
            optimizer    = optimizer,
            epsilon      = cfg.epsilon,
            step_size    = cfg.step_size,
            num_steps    = cfg.num_steps,
            alpha        = alpha,
            beta         = beta,
            epoch        = epoch,
            log_interval = cfg.log_interval,
        )

        # ── Evaluación ─────────────────────────────────────────────────────────────
        eval_stats = evaluator_fn(model, device, test_loader)

        # ── Registro de métricas ───────────────────────────────────────────────────
        metrics.update(
            epoch              = epoch,
            lambda_min         = stats.lambda_min,
            lambda_max         = stats.lambda_max,
            lambda_mean        = stats.lambda_mean,
            loss_natural       = stats.natural_loss,
            loss_robust        = stats.robust_loss,
            loss               = stats.loss,
            acc_natural        = eval_stats["natural_acc"],
            acc_robust         = eval_stats["robust_acc"],
            robust_drop        = eval_stats["robust_drop"],
            attack_success_rate= eval_stats["attack_success_rate"],
            alpha              = alpha,
            beta               = beta,
        )

        print(
            f"[Época {epoch:03d}/{cfg.epochs}] "
            f"Loss={stats.loss:.4f}  "
            f"AccNat={eval_stats['natural_acc']:.4f}  "
            f"AccRob={eval_stats['robust_acc']:.4f}  "
            f"α={alpha:.4f}  β={beta:.4f}"
        )

        # ── Checkpoint temporal / modelo final ─────────────────────────────────────
        cfg.save_checkpoints(epoch, optimizer, model)

        # ── Scheduler ──────────────────────────────────────────────────────────────
        if scheduler is not None:
            scheduler.step()

        # ── Metaheurística: propone nuevo (alpha, beta) para la próxima época ──────
        if metaheuristic is not None:
            new_alpha, new_beta = metaheuristic.update(stats.loss)
            cfg.alpha = new_alpha
            cfg.beta  = new_beta

    # ── Fin del entrenamiento: guardar métricas ─────────────────────────────────
    metrics.save_metrics()