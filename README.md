# RobustnessCNN — Entrenamiento Adversarialmente Robusto de CNNs

## Autores

- **Ademir Muñoz** — ademir.munoz.r@mail.pucv.cl
- **Jorge Villareal**

---

## Contexto

> _(Completar por los autores: motivación del trabajo, problema abordado, datasets utilizados, resultados relevantes, referencia al informe.)_

---

## Estructura del proyecto

```
RobustnessCNN/
├── main.py                  # Punto de entrada principal
├── requirements.txt         # Dependencias Python
├── .env.example             # Plantilla de configuración por variables de entorno
├── run_oceano.sbatch        # Script SLURM para ejecutar en OCEANO-PUCV
├── get.sbatch               # Script auxiliar para diagnóstico de GPU en OCEANO
├── download_datasets.sh     # Script para descargar datasets
│
├── Core/                    # Lógica central de entrenamiento y evaluación
│   ├── Config.py            # Configuración global del experimento (hiperparámetros, rutas)
│   ├── Dataconfig.py        # Carga y preprocesamiento de datasets
│   ├── train.py             # Ciclo de entrenamiento por épocas
│   ├── eval.py              # Evaluación adversarial con PGD (whitebox)
│   ├── attack.py            # Implementación del ataque PGD
│   └── Metrics.py           # Registro de métricas por batch y época (CSV)
│
├── Metodo/                  # Métodos de defensa adversarial (ver sección siguiente)
│   ├── __init__.py          # Registro centralizado de métodos
│   ├── dtrades.py           # D-TRADES / ARES — implementación Fase 2 (actual)
│   ├── dtrades_og.py        # D-TRADES original — implementación Fase 1 (referencia)
│   ├── trades.py            # TRADES (Zhang et al., 2019) — baseline
│   └── mart.py              # MART (Wang et al., 2020) — baseline
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
└── Resultado/               # Salidas generadas durante el entrenamiento
    ├── modelo/              # Pesos finales del modelo y métricas por run
    └── Temp/                # Checkpoints intermedios (cada 20 épocas por defecto)
```

### Módulo `Core/`

| Archivo | Responsabilidad |
|---|---|
| `Config.py` | Centraliza todos los hiperparámetros. Lee variables de entorno (`DATASET`, `MODEL`, `METHOD`, `EPOCHS`, etc.) y resuelve rutas de salida. |
| `Dataconfig.py` | Gestiona la carga de datasets (`cifar10`, `cifar100`, `riawelc`, etc.), augmentations (RandAugment) y DataLoaders. |
| `train.py` | Loop de entrenamiento por épocas, llamada al método de pérdida, guardado de checkpoints y métricas. |
| `eval.py` | Evaluación adversarial con PGD whitebox sobre el conjunto de test. |
| `attack.py` | Implementación genérica del ataque PGD utilizado durante entrenamiento y evaluación. |
| `Metrics.py` | Escribe `batch_metrics.csv` y `epoch_metrics.csv` en la carpeta del run. |

### Módulo `Metodo/`

Cada método expone exactamente cuatro funciones (`compute_loss`, `make_state`, `save_state`, `load_state`) que `train.py` y `main.py` llaman de forma genérica, sin conocer qué método está activo. Esto permite agregar nuevos métodos sin modificar el código de entrenamiento.

#### Métodos disponibles

| Nombre de registro | Archivo | Descripción | Fase |
|---|---|---|---|
| `d_trades` | `dtrades.py` | **D-TRADES / ARES** — método propuesto. Lambda dinámico adaptativo por clase basado en entropía local H(x) y sensibilidad S(x). | Fase 2 (actual) |
| `d_trades_og` | `dtrades_og.py` | Implementación original de D-TRADES con pesos fijos por muestra. Conservada como referencia. | Fase 1 (histórico) |
| `mart` | `mart.py` | MART (Wang et al., 2020) — baseline del estado del arte. | Baseline |
| _(registrable)_ | `trades.py` | TRADES (Zhang et al., 2019) — baseline clásico. | Baseline |

#### Nota sobre nomenclatura: ARES, D-TRADES y los archivos

El método propuesto en el informe se denomina **ARES**. En el código, su implementación activa (Fase 2) reside en `dtrades.py` y se registra bajo la clave `d_trades`, por razones históricas del desarrollo. El archivo `dtrades_og.py` corresponde a la versión de la **Fase 1** mencionada en el informe, y se conserva únicamente como referencia. **Para ejecutar ARES, usar `METHOD=d_trades`**.

### Módulo `Models/`

La función `get_model(name, num_classes, img_size)` actúa como fábrica centralizada. El parámetro `img_size` controla automáticamente el stem arquitectónico:
- `img_size ≤ 32` → stem CIFAR (conv 3×3 / stride-1), usado con CIFAR-10/100.
- `img_size > 32` → stem ImageNet (conv 7×7 / stride-2 + MaxPool), usado con RIAWELC (224px).

Los modelos disponibles son: `resnet18`, `resnet50`, `vgg10`–`vgg19`, `wideresnet` / `wrn-D-K` (ej. `wrn-34-10`).

---

## Requisitos e instalación

- Python 3.10+
- CUDA 11.8+ (recomendado para entrenamiento)

### Instalación local (venv)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Instalación en OCEANO (conda)

```bash
conda create -n robustness python=3.11
conda activate robustness
pip install -r requirements.txt
```

### Datasets

Los datasets **no se incluyen** en el repositorio. Deben colocarse bajo `./data/` con la estructura indicada arriba. Para datasets estándar (CIFAR-10, CIFAR-100) se puede usar el script auxiliar:

```bash
bash download_datasets.sh
```

Para RIAWELC, copiar manualmente la carpeta `Dataset_partitioned/` bajo `data/RIAWELC/`.

---

## Cómo ejecutarlo

Toda la configuración del experimento se gestiona a través de **variables de entorno**, ya sea mediante un archivo `.env` (ejecución local) o exportando variables directamente en el script SLURM (OCEANO).

### Variables de entorno principales

Copiar `.env.example` como `.env` y ajustar:

```bash
cp .env.example .env
```

| Variable | Descripción | Valores posibles | Default |
|---|---|---|---|
| `DATASET` | Dataset a utilizar | `cifar10`, `cifar100`, `riawelc` | `cifar10` |
| `MODEL` | Arquitectura | `resnet18`, `resnet50`, `wrn-34-10`, `vgg16`, … | `resnet18` |
| `METHOD` | Método de defensa | `d_trades`, `d_trades_og`, `mart` | `d_trades` |
| `EPOCHS` | Número de épocas | entero | `100` |
| `BATCH_SIZE` | Tamaño de batch | entero | `128` (`64` para riawelc) |
| `SEED` | Semilla de reproducibilidad | entero | `1` |
| `USE_RANDAUGMENT` | Activar RandAugment | `True` / `False` | `True` |
| `EVAL_FREQ` | Cada cuántas épocas evaluar con PGD | entero (`9999` = solo al final) | `10` |
| `WRN_DROPOUT` | Dropout para WideResNet | float (ej. `0.3`) | `0.0` |
| `TITLE` | Prefijo para la carpeta de resultados | string | _(vacío)_ |

#### Reanudación de runs interrumpidos

**Opción A** — Detecta automáticamente el checkpoint más reciente en `Resultado/Temp/<RUN_ID>/`:

```
# En .env o en run_oceano.sbatch:
RUN_ID=20260610-143000    # fecha/hora del run a continuar (sin el TITLE)
```

**Opción B** — Ruta directa a un checkpoint `.pt`:

```
CHECKPOINT_PATH=/ruta/absoluta/al/checkpoint.pt
CHECKPOINT_EPOCH=45       # época guardada en ese checkpoint
```

---

### Ejecución local

```bash
# 1. Activar entorno
source venv/bin/activate

# 2. Configurar experimento en .env (ver sección anterior)
#    Ejemplo mínimo en .env:
#      METHOD=d_trades
#      DATASET=cifar10
#      MODEL=resnet18
#      EPOCHS=100

# 3. Lanzar entrenamiento
python main.py
```

Los resultados se guardan automáticamente en:
- `Resultado/modelo/<RUN_ID>/` — modelo final (`.pt`) y métricas (`.csv`)
- `Resultado/Temp/<RUN_ID>/` — checkpoints intermedios cada 20 épocas

---

### Ejecución en OCEANO (SLURM)

Editar los hiperparámetros directamente en `run_oceano.sbatch` (sección `[6] HIPERPARÁMETROS`) y lanzar:

```bash
sbatch run_oceano.sbatch
```

#### Comandos útiles en OCEANO

```bash
squeue -u $USER                          # Ver cola de jobs propios
scancel <JOB_ID>                         # Cancelar un job
tail -f logs/<nombre>_<JOB_ID>.log       # Ver log en tiempo real
seff <JOB_ID>                            # Eficiencia del job (post-ejecución)
```

#### Recuperar resultados desde tu PC

```bash
# Todo Resultado/
scp -r usuario@oceano.pucv.cl:/ruta/proyecto/Resultado/ ./Resultado/

# Solo modelo final
scp usuario@oceano.pucv.cl:/ruta/proyecto/Resultado/modelo/**/*_final.pt ./

# Solo métricas CSV
scp usuario@oceano.pucv.cl:/ruta/proyecto/Resultado/modelo/**/epoch_metrics.csv ./
```

#### Diagnóstico de GPU

Antes de lanzar un job largo, se puede verificar el estado de la GPU con:

```bash
sbatch get.sbatch
```

Genera un log en `logs/gpu_diag_<JOB_ID>.log` con la salida de `nvidia-smi`.

---

## Resultados

Los archivos de salida por cada run se organizan bajo `Resultado/`:

| Archivo | Contenido |
|---|---|
| `Resultado/modelo/<RUN_ID>/<run_name>_final.pt` | Pesos del modelo al finalizar el entrenamiento |
| `Resultado/modelo/<RUN_ID>/<run_name>_final.tar` | Estado del optimizador al finalizar |
| `Resultado/modelo/<RUN_ID>/epoch_metrics.csv` | Métricas por época (loss, acc natural, acc robusta) |
| `Resultado/modelo/<RUN_ID>/batch_metrics.csv` | Métricas por batch (solo si `BATCH_METRICS=true`) |
| `Resultado/Temp/<RUN_ID>/*_checkpoint_<epoch>.pt` | Checkpoints intermedios cada 20 épocas |
