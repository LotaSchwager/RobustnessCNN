#!/bin/bash
#SBATCH --job-name=dtrades_cifar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dtrades_%j.log
#SBATCH --error=logs/dtrades_%j.err

# 1. Cargar módulos necesarios (Ajustar según OCEANO)
module load cuda/11.8
module load python/3.10

# 2. Crear carpetas de logs si no existen
mkdir -p logs

# 3. Entorno virtual (Si no está creado, lo crea)
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 4. Ejecución del entrenamiento
# Para usar CIFAR-100 podrías pasar un argumento o editar el script d_trades_cifar10.py
# Aquí activamos el modo DEBUG para una prueba inicial si el usuario lo desea
# export DEBUG=true 

python Entrenamiento/D_TRADES/d_trades_cifar10.py
