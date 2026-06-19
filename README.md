# RobustnessCNN — Evaluación de Robustez Adversarial (Testing)

> **Rama de pruebas.** Este repositorio contiene únicamente el código de evaluación de modelos pre-entrenados, tanto para ejecución local como en OCEANO-PUCV. No incluye código de entrenamiento.

## Autores

- **Ademir Muñoz** — ademir.munoz.r@mail.pucv.cl
- **Jorge Villareal**

---

## Estructura del proyecto

```
RobustnessCNN/
├── eval_checkpoint.py       # Punto de entrada: evalúa un checkpoint contra múltiples ataques
├── requirements.txt         # Dependencias Python
├── run_oceano.sbatch        # Script SLURM para evaluar en OCEANO-PUCV
├── download_datasets.sh     # Script para descargar datasets
│
├── Core/                    # Lógica de evaluación
│   ├── Config.py            # Configuración del experimento (hiperparámetros, rutas)
│   ├── Dataconfig.py        # Carga y preprocesamiento de datasets
│   ├── eval.py              # Evaluación adversarial con PGD (whitebox)
│   ├── attack.py            # Implementación base del ataque PGD
│   └── attacks/             # Suite de ataques para evaluación
│       ├── __init__.py      # Exporta evaluate_all_attacks y autoattack_eval
│       ├── evaluate.py      # Orquesta la evaluación con todos los ataques
│       ├── pgd.py           # Ataque PGD (Projected Gradient Descent)
│       ├── fgsm.py          # Ataque FGSM (Fast Gradient Sign Method)
│       └── autoattack_eval.py  # AutoAttack (evaluación estándar de la literatura)
│
├── Models/                  # Arquitecturas de redes neuronales
│   ├── __init__.py          # Fábrica de modelos (get_model)
│   ├── resnet.py            # ResNet-18, ResNet-50
│   ├── vgg.py               # VGG-10/11/13/16/19
│   ├── wideresnet.py        # WideResNet (WRN-28-10, WRN-34-10, WRN-16-8, etc.)
│   └── normalize.py         # Capa de normalización encapsulada (NormalizeLayer)
│
├── data/                    # Datasets (no incluidos en el repositorio)
│   ├── cifar-10-batches-py/
│   ├── cifar-100-python/
│   └── RIAWELC/
│
└── Resultado/               # Resultados de evaluaciones anteriores
    └── modelo/
        └── <RUN_ID>/
            └── metricas_evaluacion.json   # Métricas por ataque (Natural, PGD, FGSM, AutoAttack)
```

### `eval_checkpoint.py`

Script principal. Carga un modelo desde un checkpoint (`.pt`) y lo evalúa contra tres tipos de ataques: **Natural** (sin perturbación), **PGD**, **FGSM** y opcionalmente **AutoAttack**. Los resultados se guardan como `metricas_evaluacion.json` junto al checkpoint evaluado.

Se configura completamente mediante variables de entorno:

| Variable | Descripción | Default |
|---|---|---|
| `DATASET` | Dataset a evaluar | `cifar10` |
| `MODEL` | Arquitectura del modelo guardado | `resnet18` |
| `CHECKPOINT_PATH` | Ruta al archivo `.pt` a evaluar | _(requerido)_ |
| `TEST_BATCH_SIZE` | Tamaño de batch para evaluación | `32` |

### `Core/attacks/`

Suite modular de ataques adversariales. La función `evaluate_all_attacks` en `evaluate.py` los orquesta en secuencia sobre el conjunto de test y devuelve las métricas consolidadas.

### `run_oceano.sbatch`

Script SLURM adaptado para evaluación en OCEANO-PUCV. A diferencia del script de entrenamiento, solicita menos recursos (4 cores, 16 GB RAM) ya que la evaluación es considerablemente más rápida que el entrenamiento.
