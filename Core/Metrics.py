import os
import csv
import math


class Metrics:
    """
    Registra y persiste métricas de entrenamiento época a época.

    Métricas generales (todos los métodos):
        epoch, loss, loss_natural, loss_robust,
        lambda_min, lambda_max, lambda_mean,
        acc_natural, acc_robust, robust_drop, attack_success_rate


    """

    def __init__(self, results_dir: str):
        self._results_dir = results_dir
        os.makedirs(self._results_dir, exist_ok=True)

        # Métricas universales
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
    # Actualización
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
    # Persistencia en CSV
    # ------------------------------------------------------------------

    def save_metrics(self) -> None:
        """
        Guarda todas las métricas en metricas.csv dentro de results_dir.

        El archivo se abre en modo append para que las reanudaciones de
        entrenamientos interrumpidos no sobreescriban las épocas previas.
        """
        path = os.path.join(self._results_dir, "metricas.csv")
        file_exists = os.path.isfile(path)

        # Columnas base
        fieldnames = [
            "epoch",
            "loss", "loss_natural", "loss_robust",
            "lambda_min", "lambda_max", "lambda_mean",
            "acc_natural", "acc_robust",
            "robust_drop", "attack_success_rate",
        ]

        with open(path, "a", newline="") as f:
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

        print(f"[METRICS] Métricas guardadas en: {path}")

    # ------------------------------------------------------------------
    # Propiedades de lectura
    # ------------------------------------------------------------------

    @property
    def epochs_list(self):          return self._epochs_list
    @property
    def lambda_min(self):           return self._lambda_min
    @property
    def lambda_max(self):           return self._lambda_max
    @property
    def lambda_mean(self):          return self._lambda_mean
    @property
    def loss_natural(self):         return self._loss_natural
    @property
    def loss_robust(self):          return self._loss_robust
    @property
    def loss(self):                 return self._loss
    @property
    def robust_acc(self):           return self._robust_acc
    @property
    def natural_acc(self):          return self._natural_acc
    @property
    def robust_drop(self):          return self._robust_drop
    @property
    def attack_success_rate(self):  return self._attack_success_rate