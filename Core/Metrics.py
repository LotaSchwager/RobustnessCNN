import os
import csv

class Metrics:
    """
    Registra y persiste métricas de entrenamiento época a época.

    Guarda automáticamente un CSV al finalizar (llamando a save_metrics()).
    Las columnas alpha/beta permiten trazar la evolución del optimizador
    de metaheurística durante el entrenamiento.
    """

    def __init__(self, results_dir: str):
        self._results_dir = results_dir
        os.makedirs(self._results_dir, exist_ok=True)

        # -- Métricas de lambda (D-TRADES) --
        self._epochs_list: list[int] = []
        self._lambda_min:  list[float] = []
        self._lambda_max:  list[float] = []
        self._lambda_mean: list[float] = []

        # -- Pérdidas --
        self._loss_natural: list[float] = []
        self._loss_robust:  list[float] = []
        self._loss:         list[float] = []

        # -- Precisión --
        self._natural_acc: list[float] = []
        self._robust_acc:  list[float] = []
        self._robust_drop: list[float] = []
        self._attack_success_rate: list[float] = []

        # -- Parámetros de la metaheurística (por época) --
        self._alpha: list[float] = []
        self._beta:  list[float] = []

    # ------------------------------------------------------------------
    # Actualización
    # ------------------------------------------------------------------

    def update(
        self,
        epoch: int,
        lambda_min: float,
        lambda_max: float,
        lambda_mean: float,
        loss_natural: float,
        loss_robust: float,
        loss: float,
        acc_natural: float,
        acc_robust: float,
        robust_drop: float,
        attack_success_rate: float,
        alpha: float = float("nan"),
        beta:  float = float("nan"),
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
        self._alpha.append(alpha)
        self._beta.append(beta)

    # ------------------------------------------------------------------
    # Persistencia en CSV
    # ------------------------------------------------------------------

    def save_metrics(self) -> None:
        """Guarda todas las métricas en un CSV dentro de results_dir."""
        path = os.path.join(self._results_dir, "metricas.csv")
        file_exists = os.path.isfile(path)
        
        fieldnames = [
            "epoch",
            "alpha", "beta",
            "lambda_min", "lambda_max", "lambda_mean",
            "loss_natural", "loss_robust", "loss",
            "acc_natural", "acc_robust",
            "robust_drop", "attack_success_rate",
        ]
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for i in range(len(self._loss)):
                writer.writerow({
                    "epoch":               self._epochs_list[i],
                    "alpha":               self._alpha[i]               if i < len(self._alpha) else "",
                    "beta":                self._beta[i]                if i < len(self._beta)  else "",
                    "lambda_min":          self._lambda_min[i],
                    "lambda_max":          self._lambda_max[i],
                    "lambda_mean":         self._lambda_mean[i],
                    "loss_natural":        self._loss_natural[i],
                    "loss_robust":         self._loss_robust[i],
                    "loss":                self._loss[i],
                    "acc_natural":         self._natural_acc[i],
                    "acc_robust":          self._robust_acc[i],
                    "robust_drop":         self._robust_drop[i],
                    "attack_success_rate": self._attack_success_rate[i],
                })
        print(f"[METRICS] Métricas guardadas en: {path}")

    # ------------------------------------------------------------------
    # Propiedades (acceso de lectura)
    # ------------------------------------------------------------------

    @property
    def lambda_min(self):
        return self._lambda_min

    @property
    def lambda_max(self):
        return self._lambda_max

    @property
    def lambda_mean(self):
        return self._lambda_mean

    @property
    def loss_natural(self):
        return self._loss_natural

    @property
    def loss_robust(self):
        return self._loss_robust

    @property
    def loss(self):
        return self._loss

    @property
    def robust_acc(self):
        return self._robust_acc

    @property
    def natural_acc(self):
        return self._natural_acc

    @property
    def robust_drop(self):
        return self._robust_drop

    @property
    def attack_success_rate(self):
        return self._attack_success_rate

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta