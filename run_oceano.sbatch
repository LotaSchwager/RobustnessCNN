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
#SBATCH -J dtrades_cifar100
#SBATCH -p GPU
#SBATCH --qos=normal-gpu
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem=48G
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
# [5] ENTORNO CONDA
# -----------------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robustness


echo "[INFO] Python  : $(python --version)"
echo "[INFO] PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "[INFO] CUDA OK : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[INFO] GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# -----------------------------------------------------------------------------
# [6] HIPERPARÁMETROS (todos como variables de entorno)
#     El script Python lee estas variables con os.getenv().
#     Modifica aquí para cambiar el experimento sin tocar el código Python.
# -----------------------------------------------------------------------------
export DATASET="cifar100"           
export MODEL="resnet18"             
export METHOD="d_trades"
export EPOCHS=100

# Recuperación ante caídas (Resuming)
# Si tu job fue cancelado por límite de tiempo de SLURM, descomenta la siguiente
# línea y pon el nombre (fecha) exacto de tu RUN para continuar desde el Temp/:
# export RUN_ID="YYYYMMDD-HHMMSS"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[INFO] Configuración del experimento:"
echo "       Método  : $METHOD"
echo "       Dataset : $DATASET  |  Modelo: $MODEL"
echo "       Épocas  : $EPOCHS"

# -----------------------------------------------------------------------------
# [7] EJECUCIÓN
# -----------------------------------------------------------------------------
echo ""
echo "[INFO] Lanzando entrenamiento principal..."

python main.py

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
