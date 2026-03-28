import os
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from Core.Config import Config
from Core.Metrics import Metrics
from Core.Dataconfig import DataConfig, make_loaders
from Core.train import run_training
from Core.eval import eval_adv_test_whitebox_pgd

from Models.normalize import NormalizeLayer
from Models import get_model
from Metodo import get_method


def main():
    # -------------------------------------------------------------------------
    # 1) Solo las cuatro variables esenciales vienen del entorno.
    #    Todo lo demás lo resuelve Config internamente.
    # -------------------------------------------------------------------------
    dataset_name = os.getenv("DATASET", "cifar10")
    model_name   = os.getenv("MODEL",   "resnet18")
    method_name  = os.getenv("METHOD",  "d_trades")
    epochs       = int(os.getenv("EPOCHS", "100"))

    # -------------------------------------------------------------------------
    # 2) Dataset — se carga primero para conocer num_classes antes de Config
    # -------------------------------------------------------------------------
    data_cfg = DataConfig(
        name            = dataset_name,
        root            = "./data",
        batch_size      = 256,
        test_batch_size = 256,
        num_workers     = 10,
        use_cuda        = True,
        download        = False,
    )
    train_loader, test_loader, mean, std, num_classes = make_loaders(data_cfg)
    print(f"[DATA] Dataset: {dataset_name}  |  Clases: {num_classes}")

    # -------------------------------------------------------------------------
    # 3) Config — ya conoce num_classes, necesario para make_state()
    # -------------------------------------------------------------------------
    cfg = Config(
        dataset     = dataset_name,
        model       = model_name,
        method      = method_name,
        epochs      = epochs,
        num_classes = num_classes,   # necesario para que el método cree su estado
        seed        = 1,
        cuda        = True,
    )

    # -------------------------------------------------------------------------
    # 4) Método de defensa — se resuelve por nombre, igual que get_model()
    # -------------------------------------------------------------------------
    method_module   = get_method(method_name)          # módulo con compute_loss etc.
    compute_loss_fn = method_module["compute_loss"]
    make_state_fn   = method_module["make_state"]
    save_state_fn   = method_module["save_state"]
    load_state_fn   = method_module["load_state"]

    # Estado mutable del método (p.ej. ClassStats para D-TRADES).
    # Se instancia aquí, fuera de cualquier loop, para que persista todo el run.
    method_state = make_state_fn(cfg, cfg.device)
    print(f"[METHOD] {method_name} | estado inicializado")

    # -------------------------------------------------------------------------
    # 5) Modelo con normalización encapsulada
    # -------------------------------------------------------------------------
    base  = get_model(model_name, num_classes=num_classes)
    model = nn.Sequential(NormalizeLayer(mean, std), base).to(cfg.device)
    print(f"[MODEL] {model_name}  |  device: {cfg.device}")

    # -------------------------------------------------------------------------
    # 6) Optimizador y Scheduler
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
    # 7) Métricas
    # -------------------------------------------------------------------------
    metrics = Metrics(cfg.results_dir)

    # Evaluador adversarial: PGD-20 (más fuerte que el PGD-10 del entrenamiento)
    def evaluator_fn(model, device, test_loader):
        return eval_adv_test_whitebox_pgd(
            model, device, test_loader,
            epsilon   = cfg.epsilon,
            num_steps = 20,
            step_size = cfg.step_size,
        )

    # -------------------------------------------------------------------------
    # 8) Reanudación desde checkpoint
    #
    #    Opción A (RUN_ID en .env): detecta automáticamente el .pt más reciente
    #    en Resultado/Temp/<RUN_ID>/. Útil para continuar un run interrumpido
    #    en OCEANO sin tener que saber el nombre exacto del archivo.
    #
    #    Opción B (CHECKPOINT_PATH en .env): ruta directa al archivo .pt.
    #    Útil si el checkpoint está en otro directorio o si quieres partir desde
    #    un punto específico. CHECKPOINT_EPOCH indica la época guardada para
    #    que el scheduler y las métricas sean correctos.
    #    Si CHECKPOINT_EPOCH no se especifica, se intenta extraerlo del nombre.
    # -------------------------------------------------------------------------
    start_epoch = 1
    latest_cp   = None

    cp_path  = os.getenv("CHECKPOINT_PATH", "").strip()
    cp_epoch = os.getenv("CHECKPOINT_EPOCH", "").strip()

    if cp_path:
        # ── Opción B: ruta directa ────────────────────────────────────────────
        if not os.path.exists(cp_path):
            raise FileNotFoundError(f"[RESUME] CHECKPOINT_PATH no encontrado: {cp_path}")
        latest_cp = cp_path
        if cp_epoch:
            start_epoch = int(cp_epoch) + 1
        else:
            # Intenta extraer el número del nombre del archivo
            try:
                start_epoch = int(cp_path.split("_")[-1].split(".")[0]) + 1
            except ValueError:
                print("[RESUME] No se pudo extraer la época del nombre del archivo. "
                      "Define CHECKPOINT_EPOCH en .env.")
        print(f"[RESUME] Opción B — checkpoint directo: {latest_cp} (inicio época {start_epoch})")

    elif os.path.exists(cfg.temp_dir):
        # ── Opción A: detección automática en Temp/ ───────────────────────────
        pt_files = [f for f in os.listdir(cfg.temp_dir) if f.endswith(".pt")]
        if pt_files:
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
            print(f"[RESUME] Opción A — checkpoint más reciente: {latest_cp} (inicio época {start_epoch})")

    if latest_cp is not None:
        model.load_state_dict(torch.load(latest_cp, map_location=cfg.device))

        opt_cp = latest_cp.replace(".pt", ".tar")
        if os.path.exists(opt_cp):
            optimizer.load_state_dict(torch.load(opt_cp, map_location=cfg.device))
            print(f"[RESUME] Optimizer restaurado.")

        # Restaura el estado del método solo si el archivo .stats existe.
        # Con métodos que no tienen estado (make_state→None) load_state es pass
        # y el archivo nunca se crea, así que esta guarda es necesaria.
        checkpoint_base = latest_cp[:-3]   # quita ".pt"
        if os.path.exists(checkpoint_base + ".stats"):
            load_state_fn(method_state, checkpoint_base, cfg.device)

        if scheduler is not None:
            for _ in range(start_epoch - 1):
                scheduler.step()

    # -------------------------------------------------------------------------
    # 9) Ciclo de entrenamiento
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Entrenamiento  |  Método: {method_name}  |  {dataset_name}")
    print(f"  Épocas: {cfg.epochs}  |  batch: {cfg.batch}  |  lr: {cfg.lr}")
    print(f"  Resultados → {cfg.results_dir}")
    print("=" * 60 + "\n")

    run_training(
        cfg             = cfg,
        model           = model,
        optimizer       = optimizer,
        train_loader    = train_loader,
        test_loader     = test_loader,
        evaluator_fn    = evaluator_fn,
        metrics         = metrics,
        compute_loss_fn = compute_loss_fn,
        method_state    = method_state,
        save_state_fn   = save_state_fn,
        scheduler       = scheduler,
        start_epoch     = start_epoch,
    )

    print("\n[DONE] Entrenamiento finalizado.")
    print(f"       Modelo   → {cfg.results_dir}/{cfg.run_name}_final.pt")
    print(f"       Métricas → {cfg.results_dir}/metricas.csv")


if __name__ == "__main__":
    main()
