# Entrenamiento/D_TRADES/d_trades_cifar10.py

import sys
from pathlib import Path

# Asegura imports desde la raíz del proyecto
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import os
import torch
import torch.nn as nn
import torch.optim as optim

# Intenta cargar .env si existe (facilita configuración local)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from Core.Config import Config
from Core.Metrics import Metrics
from Core.Dataconfig import DataConfig, make_loaders
from Core.train import run_training
from Core.eval import eval_adv_test_whitebox_pgd

from Models.normalize import NormalizeLayer
from Models import get_model

from Metodo.dtrades import d_trades_loss
from Metaheurística import LocalSearchMetaheuristic


def main():
    # -------------------------------------------------------------------------
    # 1) Configuración base
    #    Todos los hiperparámetros se leen de variables de entorno (o del .env).
    #    Esto permite parametrizar el job de SLURM sin modificar código.
    # -------------------------------------------------------------------------
    dataset_name = os.getenv("DATASET", "cifar10")
    model_name   = os.getenv("MODEL",   "resnet18")

    cfg = Config(
        # Entrenamiento general
        batch        = int(os.getenv("BATCH_SIZE",    "128")),
        test_batch   = int(os.getenv("TEST_BATCH_SIZE","256")),
        epochs       = int(os.getenv("EPOCHS",        "100")),
        lr           = float(os.getenv("LR",          "0.1")),
        momentum     = float(os.getenv("MOMENTUM",    "0.9")),
        weight_decay = float(os.getenv("WEIGHT_DECAY","5e-4")),
        seed         = int(os.getenv("SEED",          "1")),
        cuda         = os.getenv("USE_CUDA", "true").lower() == "true",

        # D-TRADES / ataque adversarial
        epsilon   = float(os.getenv("EPSILON",    "0.0313725")),
        step_size = float(os.getenv("STEP_SIZE",  "0.0078431")),
        num_steps = int(os.getenv("NUM_STEPS",   "10")),
        alpha     = float(os.getenv("ALPHA",      "1.0")),
        beta      = float(os.getenv("BETA",       "1.0")),

        # Logging y guardado
        log_interval = int(os.getenv("LOG_INTERVAL", "100")),
        save_freq    = int(os.getenv("SAVE_FREQ",    "4")),   # cada 4 épocas → Temp/

        # Nombre identificador del experimento
        run_name = f"dtrades_{dataset_name}_{model_name}",
    )

    # -------------------------------------------------------------------------
    # 2) Modo DEBUG
    #    Reduce épocas y steps para verificar que el pipeline funciona.
    # -------------------------------------------------------------------------
    is_debug = os.getenv("DEBUG", "false").lower() == "true"
    if is_debug:
        print("[DEBUG] Modo de prueba rápida activado (2 épocas, 2 steps de ataque).")
        cfg.epochs    = 2
        cfg.num_steps = 2

    # -------------------------------------------------------------------------
    # 3) Dataset (solo local — sin descarga)
    #    Los datos deben estar en data/cifar-10 o data/cifar-100.
    #    DataConfig.root apunta a la carpeta base "./data".
    # -------------------------------------------------------------------------
    data_cfg = DataConfig(
        name            = dataset_name,
        root            = str(ROOT / "data"),   # ruta absoluta desde ROOT
        batch_size      = cfg.batch,
        test_batch_size = cfg.test_batch,
        num_workers     = int(os.getenv("NUM_WORKERS", "4")),
        use_cuda        = cfg.use_cuda,
        download        = False,                # nunca descargar (sin internet en OCEANO)
    )

    train_loader, test_loader, mean, std, num_classes = make_loaders(data_cfg)
    print(f"[DATA] Dataset: {dataset_name}  |  Clases: {num_classes}")

    # -------------------------------------------------------------------------
    # 4) Modelo (ResNet, VGG o personalizado)
    #    NormalizeLayer encapsula la normalización dentro del modelo,
    #    así los inputs adversariales también pasan por ella.
    # -------------------------------------------------------------------------
    base  = get_model(model_name, num_classes=num_classes)
    model = nn.Sequential(NormalizeLayer(mean, std), base).to(cfg.device)
    print(f"[MODEL] {model_name}  |  device: {cfg.device}")

    # -------------------------------------------------------------------------
    # 5) Optimizer + Scheduler
    # -------------------------------------------------------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr           = cfg.lr,
        momentum     = cfg.momentum,
        weight_decay = cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )

    # -------------------------------------------------------------------------
    # 6) Métricas
    # -------------------------------------------------------------------------
    metrics = Metrics(cfg.results_dir)

    # -------------------------------------------------------------------------
    # 7) Evaluador adversarial (PGD-20, más fuerte que el entrenamiento)
    # -------------------------------------------------------------------------
    def evaluator_fn(model, device, test_loader):
        return eval_adv_test_whitebox_pgd(
            model, device, test_loader,
            epsilon   = cfg.epsilon,
            num_steps = 20,        # eval más fuerte que train
            step_size = cfg.step_size,
        )

    # -------------------------------------------------------------------------
    # 8) Metaheurística para optimización dinámica de alpha y beta
    #    Cada época proporciona la pérdida como función objetivo.
    #    Para desactivarla, setear USE_METAHEURISTIC=false en el .env.
    # -------------------------------------------------------------------------
    use_meta = os.getenv("USE_METAHEURISTIC", "true").lower() == "true"
    if use_meta:
        metaheuristic = LocalSearchMetaheuristic(
            alpha0        = cfg.alpha,
            beta0         = cfg.beta,
            grid_values   = [0.0, 0.5, 1.0, 1.5, 2.0],  # cuadrícula de búsqueda
            refine_radius = 0.25,
            refine_step   = 0.125,
            lo            = 0.0,
            hi            = 3.0,
            verbose       = True,
        )
        print(f"[META] Metaheurística activada: {metaheuristic}")
    else:
        metaheuristic = None
        print(f"[META] Metaheurística desactivada. alpha={cfg.alpha}, beta={cfg.beta} (fijos)")

    # -------------------------------------------------------------------------
    # 9) Reanudación desde checkpoint (si existe en Temp/)
    # -------------------------------------------------------------------------
    latest_cp = None
    start_epoch = 1
    if os.path.exists(cfg.temp_dir):
        pt_files = [f for f in os.listdir(cfg.temp_dir) if f.endswith(".pt")]
        if pt_files:
            # Ordena por número de época (formato: runname_checkpoint_N.pt)
            def _epoch_num(fname):
                try:
                    return int(fname.split("_")[-1].split(".")[0])
                except ValueError:
                    return -1
            pt_files.sort(key=_epoch_num)
            last_ep = _epoch_num(pt_files[-1])
            if last_ep > 0:
                start_epoch = last_ep + 1
            latest_cp = os.path.join(cfg.temp_dir, pt_files[-1])
            print(f"[RESUME] Cargando modelo desde: {latest_cp} (Iniciando en época {start_epoch})")
            model.load_state_dict(
                torch.load(latest_cp, map_location=cfg.device)
            )
            opt_cp = latest_cp.replace(".pt", ".tar")
            if os.path.exists(opt_cp):
                optimizer.load_state_dict(
                    torch.load(opt_cp, map_location=cfg.device)
                )
                print(f"[RESUME] Optimizer cargado desde: {opt_cp}")
            
            # Avanzamos el scheduler para que esté en el paso correcto
            if scheduler is not None:
                for _ in range(start_epoch - 1):
                    scheduler.step()

    # -------------------------------------------------------------------------
    # 10) Entrenamiento principal
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Iniciando D-TRADES  |  {dataset_name}  |  {model_name}")
    print(f"  Épocas: {cfg.epochs}  |  batch: {cfg.batch}  |  lr: {cfg.lr}")
    print(f"  ε={cfg.epsilon:.5f}  step={cfg.step_size:.5f}  PGD-{cfg.num_steps}")
    print(f"  Resultados → {cfg.results_dir}")
    print("=" * 60 + "\n")

    run_training(
        cfg           = cfg,
        model         = model,
        optimizer     = optimizer,
        train_loader  = train_loader,
        test_loader   = test_loader,
        evaluator_fn  = evaluator_fn,
        metrics       = metrics,
        scheduler     = scheduler,
        metaheuristic = metaheuristic,
        start_epoch   = start_epoch,
    )

    print("\n[DONE] Entrenamiento finalizado.")
    print(f"       Modelo  → {cfg.results_dir}/{cfg.run_name}_final.pt")
    print(f"       Métricas → {cfg.results_dir}/metricas.csv")


if __name__ == "__main__":
    main()