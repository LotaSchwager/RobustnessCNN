#!/bin/bash
# =============================================================================
# Script para descargar y preparar los datasets localmente
# Ejecutar en el nodo de login antes de mandar el trabajo a SLURM
#
# Uso: 
#   chmod +x download_datasets.sh
#   ./download_datasets.sh
# =============================================================================

mkdir -p data
cd data

echo "========================================"
echo " Revisando CIFAR-10..."
echo "========================================"
# La extracción de cifar-10-python.tar.gz crea la carpeta 'cifar-10-batches-py'
if [ ! -d "cifar-10-batches-py" ]; then
    echo "[!] Descargando CIFAR-10 (162 MB)..."
    wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    
    echo "[!] Descomprimiendo..."
    tar -xzf cifar-10-python.tar.gz
    
    echo "[!] Limpiando archivo comprimido..."
    rm cifar-10-python.tar.gz
else
    echo "[OK] CIFAR-10 ya existe (carpeta cifar-10-batches-py encontrada)."
fi

echo ""
echo "========================================"
echo " Revisando CIFAR-100..."
echo "========================================"
# La extracción de cifar-100-python.tar.gz crea la carpeta 'cifar-100-python'
if [ ! -d "cifar-100-python" ]; then
    echo "[!] Descargando CIFAR-100 (161 MB)..."
    wget -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    
    echo "[!] Descomprimiendo..."
    tar -xzf cifar-100-python.tar.gz
    
    echo "[!] Limpiando archivo comprimido..."
    rm cifar-100-python.tar.gz
else
    echo "[OK] CIFAR-100 ya existe (carpeta cifar-100-python encontrada)."
fi

echo ""
echo "[INFO] Todos los datasets están listos en el directorio ./data/"
