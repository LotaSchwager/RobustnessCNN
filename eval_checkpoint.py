import torch
import torch.nn as nn

from Core.Config import Config
from Core.Dataconfig import DataConfig, make_loaders
from Models.normalize import NormalizeLayer
from Models import get_model

from Core.attacks import evaluate_all_attacks


def main():
    # -------------------------
    # 1) CONFIG
    # -------------------------
    dataset_name = "cifar10"
    model_name   = "resnet18"

    # ⚠️ CAMBIA ESTO por tu checkpoint real
    checkpoint_path = "d_trades_cifar10_resnet18_final(1).pt"

    # -------------------------
    # 2) DATA
    # -------------------------
    data_cfg = DataConfig(
        name="cifar10",
        root="./data",
        batch_size=128,
        test_batch_size=256,
        num_workers=4,
        use_cuda=True,
        download=False,
    )

    train_loader, test_loader, mean, std, num_classes = make_loaders(data_cfg)

    # -------------------------
    # 3) CONFIG GLOBAL
    # -------------------------
    cfg = Config(
        dataset=dataset_name,
        model=model_name,
        method="d_trades",
        epochs=1,
        num_classes=num_classes,
        cuda=True,
    )

    # -------------------------
    # 4) MODELO
    # -------------------------
    base_model = get_model(model_name, num_classes=num_classes)
    model = nn.Sequential(NormalizeLayer(mean, std), base_model).to(cfg.device)

    print(f"[INFO] Cargando modelo desde: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))

    # -------------------------
    # 5) EVALUACIÓN 🔥
    # -------------------------
    results = evaluate_all_attacks(
        model=model,
        device=cfg.device,
        test_loader=test_loader,
        cfg=cfg
    )

    print("\nResultados finales:", results)


if __name__ == "__main__":
    main()
