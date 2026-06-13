#!/bin/bash
#SBATCH -J gpu_diag
#SBATCH -p GPU
#SBATCH --qos=normal-gpu
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH --output=logs/gpu_diag_%j.log
#SBATCH --error=logs/gpu_diag_%j.err

echo "======================================================"
echo "  NODO     : $(hostname)"
echo "  FECHA    : $(date)"
echo "======================================================"

echo ""
echo "--- nvidia-smi (resumen) ---"
nvidia-smi

echo ""
echo "--- Procesos en la GPU ---"
nvidia-smi pmon -c 1

echo ""
echo "--- Todos los jobs corriendo en este nodo ---"
squeue --nodelist=$(hostname) -o "%.10i %.10u %.10T %.6b %R"
