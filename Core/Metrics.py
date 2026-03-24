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

    Métricas específicas de D-TRADES (opcionales, se incluyen si se pasan):
        alpha_c_0 … alpha_c_N  — alpha por clase (N = num_classes)
        beta_c_0  … beta_c_N   — beta  por clase (N = num_classes)

    Las columnas de alpha/beta por clase se añaden dinámicamente en el
    primer update() que las reciba, por lo que el CSV es siempre correcto
    aunque el método activo no las use.
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

        # Métricas por clase (D-TRADES) — listas de listas
        # Cada entrada es una lista de num_classes valores para esa época.
        # Si el método no las pasa, permanecen vacías y no aparecen en el CSV.
        self._alpha_per_class: list[list[float]] = []
        self._beta_per_class:  list[list[float]] = []
        self._num_classes: int = 0   # se descubre en el primer update con datos

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
        # D-TRADES: alpha y beta por clase (lista de num_classes floats)
        # Si el método no los usa, se omiten sin problema.
        alpha_per_class: "list[float] | None" = None,
        beta_per_class:  "list[float] | None" = None,
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

        # alpha/beta por clase (solo D-TRADES)
        if alpha_per_class is not None:
            self._alpha_per_class.append(list(alpha_per_class))
            nc = len(alpha_per_class)
            if self._num_classes == 0:
                self._num_classes = nc
        else:
            # Marcador vacío para mantener alineación de filas
            self._alpha_per_class.append([])

        if beta_per_class is not None:
            self._beta_per_class.append(list(beta_per_class))
        else:
            self._beta_per_class.append([])

    # ------------------------------------------------------------------
    # Persistencia en CSV
    # ------------------------------------------------------------------

    def save_metrics(self) -> None:
        """
        Guarda todas las métricas en metricas.csv dentro de results_dir.

        Si alpha/beta por clase fueron registrados, el CSV incluirá las
        columnas  alpha_c_0, alpha_c_1, …, beta_c_0, beta_c_1, …
        con tantas columnas como clases haya.

        El archivo se abre en modo append para que las reanudaciones de
        entrenamientos interrumpidos no sobreescriban las épocas previas.
        """
        path = os.path.join(self._results_dir, "metricas.csv")
        file_exists = os.path.isfile(path)

        nc = self._num_classes

        # Columnas base
        fieldnames = [
            "epoch",
            "loss", "loss_natural", "loss_robust",
            "lambda_min", "lambda_max", "lambda_mean",
            "acc_natural", "acc_robust",
            "robust_drop", "attack_success_rate",
        ]
        # Columnas por clase (D-TRADES) — solo si hay datos
        if nc > 0:
            fieldnames += [f"alpha_c_{c}" for c in range(nc)]
            fieldnames += [f"beta_c_{c}"  for c in range(nc)]

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
                # Columnas por clase (pueden estar vacías si el método no las usa)
                alpha_row = self._alpha_per_class[i] if i < len(self._alpha_per_class) else []
                beta_row  = self._beta_per_class[i]  if i < len(self._beta_per_class)  else []
                for c in range(nc):
                    row[f"alpha_c_{c}"] = alpha_row[c] if c < len(alpha_row) else ""
                    row[f"beta_c_{c}"]  = beta_row[c]  if c < len(beta_row)  else ""

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
    @property
    def alpha_per_class(self):      return self._alpha_per_class
    @property
    def beta_per_class(self):       return self._beta_per_class
    @property
    def num_classes(self):          return self._num_classes