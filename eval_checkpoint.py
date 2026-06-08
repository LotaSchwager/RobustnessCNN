import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Core.Dataconfig import DataConfig, make_datasets
from Models.normalize import NormalizeLayer
from Models import get_model

from Core.attacks import evaluate_all_attacks


def fixed_forward_resnet(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    if hasattr(self, 'maxpool') and self.maxpool is not None:
        out = self.maxpool(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out


def main():
    # -------------------------------------------------------------------------
    # 1) CONFIGURACIÓN DESDE ENTORNO
    # -------------------------------------------------------------------------
    dataset_name = "riawelc"
    model_name   = os.getenv("MODEL", "resnet18").lower()

    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CHECKPOINT_PATH no proporcionado o no existe: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hiperparámetros de evaluación (mismos que entrenamiento RIAWELC)
    cfg = types.SimpleNamespace(
        device=device,
        epsilon=8/255,
        step_size=2/255,
    )

    # -------------------------------------------------------------------------
    # 2) CARGA DE DATOS (solo test loader, sin crear train loader)
    # -------------------------------------------------------------------------
    data_cfg = DataConfig(
        name=dataset_name,
        root="./data",
        test_batch_size=1,
        num_workers=4,
        use_cuda=True,
        download=False,
    )

    _, test_ds, mean, std, num_classes = make_datasets(data_cfg)

    kwargs = {"num_workers": data_cfg.num_workers, "pin_memory": True} if data_cfg.use_cuda else {}
    test_loader = DataLoader(test_ds, batch_size=data_cfg.test_batch_size, shuffle=False, **kwargs)

    print("\n========== DEBUG DATASET ==========")
    print("Número de clases:", num_classes)
    print("Clases detectadas:", getattr(test_ds, 'classes', 'No disponibles'))
    print("Cantidad test:", len(test_ds))
    print("==================================\n")

    # -------------------------------------------------------------------------
    # 3) MODELO
    # -------------------------------------------------------------------------
    print(f"[INFO] Inicializando arquitectura '{model_name}' para {num_classes} clases...")

    is_wideresnet = "wideresnet" in model_name or model_name.startswith("wrn")

    if is_wideresnet:
        base_model = get_model(model_name, num_classes=num_classes, img_size=224, dropout_rate=0.3)
        print("[INFO] WideResNet configurado nativamente para 224x224.")
    else:
        base_model = get_model(model_name, num_classes=num_classes)
        base_model.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        base_model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        base_model.forward = types.MethodType(fixed_forward_resnet, base_model)
        print("[INFO] ResNet configurada con parche conv1/maxpool para 224x224.")

    model = nn.Sequential(NormalizeLayer(mean, std), base_model).to(device)

    # -------------------------------------------------------------------------
    # 4) CARGA DEL CHECKPOINT
    # -------------------------------------------------------------------------
    print(f"[INFO] Cargando modelo desde: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    # Eliminar prefijo 'module.' de entrenamientos multi-GPU
    fixed_state_dict = {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(fixed_state_dict, strict=True)
    print("[SUCCESS] Pesos cargados de manera estricta y exitosa.")

    # -------------------------------------------------------------------------
    # 5) EVALUACIÓN
    # -------------------------------------------------------------------------
    model.eval()
    torch.cuda.empty_cache()

    print(f"\n[INFO] Iniciando suite de ataques adversariales en {device}...")
    evaluate_all_attacks(
        model=model,
        device=device,
        test_loader=test_loader,
        cfg=cfg
    )


if __name__ == "__main__":
    main()
