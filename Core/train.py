from __future__ import print_function
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from typing import Any, Callable, Dict, Optional
from Metodo.dtrades import d_trades_loss

@dataclass
class TrainStats:
    loss: float
    natural_loss: float
    robust_loss: float
    lambda_min: float
    lambda_max: float
    lambda_mean: float

def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader,
    optimizer: torch.optim.Optimizer,
    epsilon: float,
    step_size: float,
    num_steps: int,
    alpha: float,
    beta: float,
    epoch: int,
    log_interval: int = 100,
    ) -> TrainStats:
    """
    Entrena el modelo durante una época.
    """
    model.train()
    # Contadores
    seen = 0
    epoch_loss = 0
    epoch_natural_loss = 0
    epoch_robust_loss = 0
    # Listas para medir lambda
    lambda_min = []
    lambda_max = []
    lambda_mean = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)

        # calculate robust loss
        loss, lambda_value, loss_natural, loss_robust_dynamic = d_trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=step_size,
                           epsilon=epsilon,
                           perturb_steps=num_steps,
                           alpha=alpha,
                           beta=beta,
                        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        bs = data.size(0)
        epoch_loss += loss.item() * bs
        epoch_natural_loss += loss_natural.item() * bs
        epoch_robust_loss += loss_robust_dynamic.item() * bs
        seen += bs

        lambda_min.append(lambda_value.min().item())
        lambda_max.append(lambda_value.max().item())
        lambda_mean.append(lambda_value.mean().item())

        # Print progress
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tLambda Mean: {:.2f} \tLambda Min: {:.2f} \tLambda Max: {:.2f}'.format(
                                            epoch,
                                            batch_idx * len(data),
                                            len(train_loader.dataset),
                                            100. * batch_idx / len(train_loader),
                                            loss.item(),
                                            np.mean(lambda_mean),
                                            np.min(lambda_min),
                                            np.max(lambda_max)
                                            )
            )
        
        # Modo DEBUG: salir después de 5 batches
        if os.getenv("DEBUG", "false").lower() == "true" and batch_idx >= 4:
            break
    return TrainStats(
        loss=epoch_loss / seen,
        natural_loss=epoch_natural_loss / seen,
        robust_loss=epoch_robust_loss / seen,
        lambda_min=np.min(lambda_min),
        lambda_max=np.max(lambda_max),
        lambda_mean=np.mean(lambda_mean),
    )

def run_training(
    cfg,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    test_loader,
    evaluator_fn: Callable[[nn.Module, torch.device, Any], Dict[str, float]], 
    metrics,
    scheduler: Optional[Any] = None,
) -> None:
    """
    Entrena el modelo durante el número de épocas especificado.
    """
    device = cfg.device

    for epoch in range(1, cfg.epochs + 1):
        stats = train_one_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epsilon=cfg.epsilon,
            step_size=cfg.step_size,
            num_steps=cfg.num_steps,
            alpha=cfg.alpha,
            beta=cfg.beta,
            epoch=epoch,
            log_interval=cfg.log_interval,
        )

        eval_stats = evaluator_fn(model, device, test_loader)

        metrics.update(
            stats.lambda_min,
            stats.lambda_max,
            stats.lambda_mean,
            stats.natural_loss,
            stats.robust_loss,
            stats.loss,
            eval_stats['natural_acc'],
            eval_stats['robust_acc'],
            eval_stats['robust_drop'],
            eval_stats['attack_success_rate'],
        )

        cfg.save_checkpoints(epoch, optimizer, model)

        if scheduler is not None:
            scheduler.step()