#!/bin/bash
# =============================================================================
#  SLURM job script para OCEANO-PUCV  |  Proyecto de Título
#  Descripción: Entrenamiento CNN con D-TRADES (con metaheurística) sobre CIFAR-10/100
#
#  Uso:
#    sbatch run_oceano.sbatch
#
#  Salidas principales:
#    logs/<nombre>_<JOBID>.log      → log completo del job
#    Resultado/modelo/<RUN_ID>/     → pesos finales del modelo (.pt) + optimizer (.tar)
#    Resultado/modelo/<RUN_ID>/metricas.csv  → métricas por época (loss, acc, alpha, beta…)
#    Resultado/Temp/<RUN_ID>/       → checkpoints intermedios cada SAVE_FREQ épocas
#
#  Para recuperar los resultados desde tu PC:
#    scp -r usuario@oceano.pucv.cl:/ruta/proyecto/Resultado/ ./Resultado/
# =============================================================================

# -----------------------------------------------------------------------------
# [1] CONFIGURACIÓN DEL JOB
#     Ajusta estos valores según tu experimento.
# -----------------------------------------------------------------------------
#SBATCH --J dtrades_cifar100
#SBATCH -p GPU
#SBATCH --qos=normal-gpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
##SBATCH --time=48:00:00

# -----------------------------------------------------------------------------
# [2] LOGS
#     %x = nombre del job, %j = ID del job.
#     Los logs se crean en ./logs/ relativo al directorio de lanzamiento.
# -----------------------------------------------------------------------------
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Notificación por correo (descomenta y configura si quieres recibirla)
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=tu.correo@pucv.cl

# =============================================================================
# INICIO
# =============================================================================

echo "======================================================"
echo "  JOB ID   : $SLURM_JOB_ID"
echo "  JOB NAME : $SLURM_JOB_NAME"
echo "  NODO     : $(hostname)"
echo "  INICIO   : $(date)"
echo "======================================================"

# -----------------------------------------------------------------------------
# [3] MÓDULOS
#     Ajusta las versiones según lo disponible en OCEANO.
#     Usa `module avail` en el nodo de login para ver qué hay disponible.
# -----------------------------------------------------------------------------
module purge
module load cuda/12.6
module load python/3.10

echo "[INFO] Módulos cargados:"
module list

# -----------------------------------------------------------------------------
# [4] ESTRUCTURA DE CARPETAS
#     Se crean aquí para que SLURM pueda escribir los logs desde el inicio.
# -----------------------------------------------------------------------------
mkdir -p logs
mkdir -p Resultado/modelo
mkdir -p Resultado/Temp

echo "[INFO] Estructura de carpetas lista."

# -----------------------------------------------------------------------------
# [5] ENTORNO VIRTUAL
#     NOTA MÁXIMA IMPORTANCIA: Los nodos GPU de OCEANO NO tienen internet.
#     Debes crear el entorno virtual e instalar los requirements desde el
#     nodo de login (head node) ANTES de enviar este job con sbatch.
#     Comandos a ejecutar una sola vez en el nodo de login:
#       python3 -m venv venv
#       source venv/bin/activate
#       pip install -r requirements.txt
# -----------------------------------------------------------------------------
if [ ! -d "venv" ]; then
    echo "[ERROR] No se encontró el directorio 'venv'."
    echo "[ERROR] Crea e instala las dependencias en el nodo de login primero."
    exit 1
else
    echo "[INFO] Activando entorno virtual..."
    source venv/bin/activate
fi


echo "[INFO] Python  : $(python --version)"
echo "[INFO] PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "[INFO] CUDA OK : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[INFO] GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# -----------------------------------------------------------------------------
# [6] HIPERPARÁMETROS (todos como variables de entorno)
#     El script Python lee estas variables con os.getenv().
#     Modifica aquí para cambiar el experimento sin tocar el código Python.
# -----------------------------------------------------------------------------
export DATASET="cifar100"            # cifar10 | cifar100
export MODEL="resnet18"             # resnet18 | resnet50 | vgg16 | ...
export EPOCHS=10
export BATCH_SIZE=128
export TEST_BATCH_SIZE=256
export LR=0.1
export MOMENTUM=0.9
export WEIGHT_DECAY=5e-4
export SEED=1
export NUM_WORKERS=4

# Ataque adversarial (epsilon = 8/255, step = 2/255)
export EPSILON=0.031372549
export STEP_SIZE=0.007843137
export NUM_STEPS=10

# D-TRADES: valores iniciales de alpha y beta
# (la metaheurística los optimiza durante el entrenamiento)
export ALPHA=1.0
export BETA=1.0

# Guardado y logging
export SAVE_FREQ=4                  # guarda checkpoint en Temp/ cada 4 épocas
export LOG_INTERVAL=100

# Metaheurística  (true: activa la optimización dinámica de alpha/beta)
export USE_METAHEURISTIC=true

# Recuperación ante caídas (Resuming)
# Si tu job fue cancelado por límite de tiempo de SLURM, descomenta la siguiente
# línea y pon el nombre (fecha) exacto de tu RUN para continuar desde el Temp/:
# export RUN_ID="YYYYMMDD-HHMMSS"

# En OCEANO los nodos GPU no tienen internet → NUNCA cambiar esto a true
export DOWNLOAD=false
export DEBUG=false
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[INFO] Configuración del experimento:"
echo "       Dataset : $DATASET  |  Modelo: $MODEL"
echo "       Épocas  : $EPOCHS   |  Batch : $BATCH_SIZE"
echo "       α₀=$ALPHA  β₀=$BETA  |  Metaheurística: $USE_METAHEURISTIC"

# -----------------------------------------------------------------------------
# [7] EJECUCIÓN
# -----------------------------------------------------------------------------
echo ""
echo "[INFO] Lanzando entrenamiento D-TRADES..."

python Entrenamiento/D_TRADES/d_trades_cifar10.py

EXIT_CODE=$?

# -----------------------------------------------------------------------------
# [8] CIERRE
# -----------------------------------------------------------------------------
echo ""
echo "======================================================"
echo "  FIN      : $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ESTADO   : ÉXITO ✓"
    echo ""
    echo "  Archivos de salida:"
    echo "    Modelos    → Resultado/modelo/"
    echo "    Checkpoints → Resultado/Temp/"
    echo "  Para copiar a tu PC:"
    echo "    scp -r $(whoami)@oceano.pucv.cl:$(pwd)/Resultado/ ./Resultado/"
else
    echo "  ESTADO   : ERROR (código $EXIT_CODE) ✗"
    echo "  Revisa el log en: logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi
echo "======================================================"

exit $EXIT_CODE

# =============================================================================
# REFERENCIA RÁPIDA
# =============================================================================
#
#  COMANDOS ÚTILES EN OCEANO:
#  -------------------------------------------------
#  Enviar job            →  sbatch run_oceano.sbatch
#  Ver tu cola           →  squeue -u $USER
#  Cancelar job          →  scancel <JOB_ID>
#  Info del job          →  scontrol show job <JOB_ID>
#  Eficiencia post-job   →  seff <JOB_ID>
#  Ver módulos           →  module avail
#  Log en tiempo real    →  tail -f logs/<nombre>_<JOB_ID>.log
#
#  RECUPERAR RESULTADOS (desde tu PC):
#  -------------------------------------------------
#  Todo Resultado/       →  scp -r usuario@oceano.pucv.cl:/ruta/Resultado/ ./
#  Solo modelo final     →  scp usuario@oceano.pucv.cl:/ruta/Resultado/modelo/**/*_final.pt ./
#  Solo métricas CSV     →  scp usuario@oceano.pucv.cl:/ruta/Resultado/modelo/**/metricas.csv ./
#
#  CONDA (alternativa al venv):
#  -------------------------------------------------
#  module load anaconda/2023
#  conda activate mi_entorno
#  (eliminar sección [5])
#
#  MÚLTIPLES EXPERIMENTOS (array jobs):
#  -------------------------------------------------
#  #SBATCH --array=1-5
#  export SEED=$SLURM_ARRAY_TASK_ID   # 5 semillas distintas
#
# =============================================================================
