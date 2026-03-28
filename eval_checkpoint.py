import os
import json
import torch
import torch.nn as nn

# Las variables de entorno se leen directamente desde el sistema (exportadas en el .sbatch)


from Core.Config import Config
from Core.Dataconfig import DataConfig, make_loaders
from Models.normalize import NormalizeLayer
from Models import get_model

from Core.attacks import evaluate_all_attacks


def main():
    # -------------------------------------------------------------------------
    # 1) CONFIG DESDE ENTORNO
    # -------------------------------------------------------------------------
    dataset_name = os.getenv("DATASET", "cifar10")
    model_name   = os.getenv("MODEL",   "resnet18")
    method_name  = os.getenv("METHOD",  "d_trades")
    
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CHECKPOINT_PATH no proporcionado o no existe: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 2) DATA (solo testeo)
    # -------------------------------------------------------------------------
    data_cfg = DataConfig(
        name            = dataset_name,
        root            = "./data",
        batch_size      = 128,          
        test_batch_size = 256,
        num_workers     = 4,
        use_cuda        = True,
        download        = False,
    )

    _, test_loader, mean, std, num_classes = make_loaders(data_cfg)
    print(f"[DATA] Dataset: {dataset_name}  |  Clases: {num_classes}")

    # -------------------------------------------------------------------------
    # 3) CONFIG GLOBAL
    # -------------------------------------------------------------------------
    cfg = Config(
        dataset     = dataset_name,
        model       = model_name,
        method      = method_name,
        epochs      = 1,             # irrelevante para eval
        num_classes = num_classes,   
        seed        = 1,
        cuda        = True,
    )

    # -------------------------------------------------------------------------
    # 4) MODELO
    # -------------------------------------------------------------------------
    base_model = get_model(model_name, num_classes=num_classes)
    model = nn.Sequential(NormalizeLayer(mean, std), base_model).to(cfg.device)

    print(f"[INFO] Cargando modelo desde: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))

    # -------------------------------------------------------------------------
    # 5) EVALUACIÓN MULTI-ATAQUE
    # -------------------------------------------------------------------------
    print("\n[INFO] Iniciando evaluacion de ataques (Natural, PGD, FGSM, AutoAttack)...")
    results = evaluate_all_attacks(
        model=model,
        device=cfg.device,
        test_loader=test_loader,
        cfg=cfg
    )

    # -------------------------------------------------------------------------
    # 6) GUARDAR RESULTADOS (Métricas)
    # -------------------------------------------------------------------------
    results_dir = os.path.dirname(checkpoint_path)
    if not results_dir:
        results_dir = "."
    output_file = os.path.join(results_dir, "metricas_evaluacion.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n[INFO] Evaluacion finalizada. Resultados guardados en: {output_file}")


if __name__ == "__main__":
    main()
