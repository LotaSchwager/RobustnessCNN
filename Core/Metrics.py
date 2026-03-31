import os
import csv
import math


class Metrics:
    """
    Registra y persiste métricas de entrenamiento.
    - Época a época: metricas.csv
    - Batch a batch: metricas_batch.csv
    """

    def __init__(self, results_dir: str):
        self._results_dir = results_dir
        os.makedirs(self._results_dir, exist_ok=True)

        self._epoch_file = os.path.join(self._results_dir, "metricas.csv")
        self._batch_file = os.path.join(self._results_dir, "metricas_batch.csv")

        # Variables almacenadas en memoria antes de hacer flush a disco (por época)
        self._epochs_list:        list[int]   = []
        self._lambda_min:         list[float] = []
        self._lambda_max:         list[float] = []
        self._lambda_mean:        list[float] = []
        self._loss_natural:       list[float] = []
        self._loss_robust:        list[float] = []
        self._loss:               list[float] = []
        self._natural_acc:        list[float] = []
        self._robust_acc:         list[float] = []
        self._robust_drop:        list[float] = []
        self._attack_success_rate:list[float] = []

    # ------------------------------------------------------------------
    # Actualización Época a Época
    # ------------------------------------------------------------------
    def update(
        self,
        epoch:               int,
        loss:                float,
        loss_natural:        float        = 0.0,
        loss_robust:         float        = 0.0,
        lambda_min:          float        = math.nan,
        lambda_max:          float        = math.nan,
        lambda_mean:         float        = math.nan,
        acc_natural:         float        = 0.0,
        acc_robust:          float        = 0.0,
        robust_drop:         float        = 0.0,
        attack_success_rate: float        = 0.0,
    ) -> None:
        """
        Almacena métricas de una sola época en memoria temporal hasta el próximo save_metrics().
        """
        self._epochs_list.append(epoch)
        self._lambda_min.append(lambda_min)
        self._lambda_max.append(lambda_max)
        self._lambda_mean.append(lambda_mean)
        self._loss_natural.append(loss_natural)
        self._loss_robust.append(loss_robust)
        self._loss.append(loss)
        self._natural_acc.append(acc_natural)
        self._robust_acc.append(acc_robust)
        self._robust_drop.append(robust_drop)
        self._attack_success_rate.append(attack_success_rate)

    # ------------------------------------------------------------------
    # Actualización Batch a Batch
    # ------------------------------------------------------------------
    def update_batch(self, epoch: int, batch_stats: list[dict]) -> None:
        """
        Persiste directamente a 'metricas_batch.csv' sin guardarlo en memoria.
        `batch_stats` debe ser una lista de diccionarios, cada uno con al menos:
        'iteration', 'batch_idx', 'loss', 'loss_natural', 'loss_robust',
        'lambda_min', 'lambda_mean', 'lambda_max'
        """
        file_exists = os.path.isfile(self._batch_file)
        fieldnames = [
            "epoch", "iteration", "batch_idx", 
            "loss", "loss_natural", "loss_robust", 
            "lambda_min", "lambda_mean", "lambda_max"
        ]

        with open(self._batch_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for stat in batch_stats:
                row = {"epoch": epoch}
                row.update(stat)  # Agrega la info de su diccionario interno
                writer.writerow(row)

    # ------------------------------------------------------------------
    # Persistencia en CSV
    # ------------------------------------------------------------------
    def save_metrics(self) -> None:
        """
        Guarda el buffer actual de métricas de época a 'metricas.csv' y limpia 
        el buffer para evitar duplicación.
        """
        file_exists = os.path.isfile(self._epoch_file)
        fieldnames = [
            "epoch",
            "loss", "loss_natural", "loss_robust",
            "lambda_min", "lambda_max", "lambda_mean",
            "acc_natural", "acc_robust",
            "robust_drop", "attack_success_rate",
        ]

        with open(self._epoch_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for i in range(len(self._loss)):
                row: dict = {
                    "epoch":               self._epochs_list[i],
                    "loss":                self._loss[i],
                    "loss_natural":        self._loss_natural[i],
                    "loss_robust":         self._loss_robust[i],
                    "lambda_min":          self._lambda_min[i],
                    "lambda_max":          self._lambda_max[i],
                    "lambda_mean":         self._lambda_mean[i],
                    "acc_natural":         self._natural_acc[i],
                    "acc_robust":          self._robust_acc[i],
                    "robust_drop":         self._robust_drop[i],
                    "attack_success_rate": self._attack_success_rate[i],
                }
                writer.writerow(row)
                
        # Flush de buffers
        self._epochs_list.clear()
        self._lambda_min.clear()
        self._lambda_max.clear()
        self._lambda_mean.clear()
        self._loss_natural.clear()
        self._loss_robust.clear()
        self._loss.clear()
        self._natural_acc.clear()
        self._robust_acc.clear()
        self._robust_drop.clear()
        self._attack_success_rate.clear()

        print(f"[METRICS] Guardadas con éxito")