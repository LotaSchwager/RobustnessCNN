# Entrenamiento/D_TRADES/d_trades_cifar10.py

import sys
from pathlib import Path

# Asegura imports desde la raíz del proyecto
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from Config import Config
from Metrics import Metrics

from core.Dataconfig import DataConfig, make_loaders
from core.train import run_training
from core.eval import eval_adv_test_whitebox_pgd

from Models.normalize import NormalizeLayer
from Models.resnet import ResNet18

from Metodo.dtrades import d_trades_loss


def main():
    # -------------------------
    # 1) Config base (alpha=beta=1.0)
    # -------------------------
    cfg = Config(
        # entrenamiento
        batch=128,
        test_batch=256,
        epochs=100,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        seed=1,
        cuda=True,

        # D-TRADES / ataque
        epsilon=8/255,
        step_size=2/255,
        num_steps=10,      # pasos para generar adversarial en training (puedes subir)
        alpha=1.0,
        beta=1.0,

        # logging / guardado
        log_interval=100,
        save_freq=5,
    )

    # -------------------------
    # 2) Data (CIFAR-10)
    # -------------------------
    data_cfg = DataConfig(
        name="cifar10",
        root="./data",
        batch_size=cfg.batch,
        test_batch_size=cfg.test_batch,
        num_workers=4,
        use_cuda=cfg.use_cuda,
        download=True,
    )

    train_loader, test_loader, mean, std, num_classes = make_loaders(data_cfg)

    # -------------------------
    # 3) Modelo (NormalizeLayer dentro del modelo)
    # -------------------------
    base = ResNet18(num_classes=num_classes)  # asegúrate que ResNet18 acepte num_classes
    model = nn.Sequential(NormalizeLayer(mean, std), base).to(cfg.device)

    # -------------------------
    # 4) Optimizer (+ scheduler opcional)
    # -------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    # Scheduler opcional (ejemplo MultiStep)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[75, 90],
        gamma=0.1
    )

    # -------------------------
    # 5) Métricas
    # -------------------------
    metrics = Metrics(cfg.results_dir)

    # -------------------------
    # 6) Evaluator (usa PGD)
    # -------------------------
    def evaluator_fn(model, device, test_loader):
        # Si quieres separar pasos de eval vs train, aquí puedes usar otros valores.
        return eval_adv_test_whitebox_pgd(
            model, device, test_loader,
            epsilon=cfg.epsilon,
            num_steps=20,        # típico evaluar más fuerte que entrenar
            step_size=cfg.step_size,
        )

    # -------------------------
    # 7) Entrenamiento principal
    # -------------------------
    run_training(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        d_trades_loss_fn=d_trades_loss,
        evaluator_fn=evaluator_fn,
        metrics=metrics,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()