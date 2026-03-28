#!/bin/bash
# =============================================================================
#  SLURM job script para OCEANO-PUCV  |  Proyecto de Título
#  Descripción: Evaluación de CNN con métricas de robustez (Natural, PGD, FGSM, AutoAttack)
#
#  Uso:
#    sbatch run_oceano.sbatch
#
#  Salidas principales:
#    logs/<nombre>_<JOBID>.log      → log completo del job
#    <Ruta_del_Checkpoint>/metricas_evaluacion.json  → métricas finales guardadas
#
#  Para recuperar los resultados desde tu PC:
#    scp -r usuario@oceano.pucv.cl:/ruta/proyecto/Resultado/ ./Resultado/
# =============================================================================

# -----------------------------------------------------------------------------
# [1] CONFIGURACIÓN DEL JOB
#     Ajusta estos valores según tu configuración y prioridad para la evaluación.
# -----------------------------------------------------------------------------
#SBATCH --J eval_cnn
#SBATCH -p GPU
#SBATCH --qos=normal-gpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
##SBATCH --time=12:00:00

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
#     Se crea aquí para que SLURM pueda escribir los logs desde el inicio.
# -----------------------------------------------------------------------------
mkdir -p logs

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
# [6] HIPERPARÁMETROS DE EVALUACIÓN (variables de entorno)
#     El script Python de evaluación lee estas variables para configurar el modelo
#     y cargar los pesos desde tu CHECKPOINT_PATH.
# -----------------------------------------------------------------------------
export DATASET="cifar10"
export MODEL="resnet18"
export METHOD="d_trades"

# INGRESA AQUÍ LA RUTA ABSOLUTA O RELATIVA DEL MODELO A EVALUAR
export CHECKPOINT_PATH="/home/ademir.munoz/RobustnessCNN/Resultado/modelo/"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[INFO] Configuración de la Evaluación:"
echo "       Método      : $METHOD"
echo "       Dataset     : $DATASET  |  Modelo: $MODEL"
echo "       Checkpoint  : $CHECKPOINT_PATH"

# -----------------------------------------------------------------------------
# [7] EJECUCIÓN
# -----------------------------------------------------------------------------
echo ""
echo "[INFO] Lanzando pipeline de pruebas y métricas..."

python eval_checkpoint.py

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
    echo "    Las métricas de Acc natural, PGD, FGSM y AutoAttack se han guardado"
    echo "    junto al checkpoint en el archivo 'metricas_evaluacion.json'."
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
# =============================================================================
