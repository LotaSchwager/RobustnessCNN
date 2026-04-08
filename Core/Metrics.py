import os
import csv
import math
import numpy as np


def _safe_corrcoef(a, b):
    """Correlación de Pearson segura; devuelve NaN si algún vector es constante."""
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _stats(arr):
    """Devuelve dict con std, mean, min, max de un array numpy 1-D."""
    if len(arr) == 0:
        return {"std": float("nan"), "mean": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "std":  float(np.std(arr)),
        "mean": float(np.mean(arr)),
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
    }


class Metrics:
    """
    Sistema de métricas detallado para D-TRADES.

    Genera dos archivos CSV:
      • batch_metrics.csv  — una fila por cada batch (granularidad fina).
      • epoch_metrics.csv  — una fila por época (promedios de los batches).

    Métricas registradas:
      - Lambda (peso dinámico):          std, mean, min, max
      - H  (entropía normalizada):       std, mean, min, max
      - S  (sensibilidad normalizada):   std, mean, min, max
      - Correlaciones:  corr(λ, S),  corr(λ, H)
      - Métricas por tipo de muestra:    lam_correct, lam_incorrect
      - loss_natural, loss_robust
    """

    # Columnas en orden para ambos CSV
    _FIELDNAMES = [
        "identifier",
        # Lambda final
        "lam_std", "lam_mean", "lam_min", "lam_max",
        # Entropía
        "H_std", "H_mean", "H_min", "H_max",
        # Sensibilidad
        "S_std", "S_mean", "S_min", "S_max",
        # Correlaciones de lambda con componentes
        "corr_lam_sensitivity", "corr_lam_entropy",
        # Métricas por tipo de muestra
        "lam_correct_mean", "lam_incorrect_mean",
        # Losses
        "loss_natural", "loss_robust",
    ]

    def __init__(self, results_dir: str):
        self._results_dir = results_dir
        os.makedirs(self._results_dir, exist_ok=True)

        self._batch_csv = os.path.join(self._results_dir, "batch_metrics.csv")
        self._epoch_csv = os.path.join(self._results_dir, "epoch_metrics.csv")

        # Escribir headers si los archivos no existen
        for path in (self._batch_csv, self._epoch_csv):
            if not os.path.isfile(path):
                with open(path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=self._FIELDNAMES).writeheader()

        # Acumuladores para el promedio por época
        self._epoch_batches: list[dict] = []
        self._current_epoch: int = 0

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def record_batch(self, epoch: int, batch_idx: int, info: dict) -> None:
        """
        Registra las métricas de un solo batch y las escribe al CSV de batch.

        Parámetros
        ----------
        epoch     : número de la época actual.
        batch_idx : índice del batch dentro de la época.
        info      : dict devuelto por dtrades.compute_loss con arrays numpy.
        """
        # Si cambió la época, resetear acumuladores
        if epoch != self._current_epoch:
            self._epoch_batches = []
            self._current_epoch = epoch

        row = self._compute_row(
            identifier=f"batch_{batch_idx}_epoca_{epoch}",
            info=info,
        )

        self._epoch_batches.append(row)
        self._write_row(self._batch_csv, row)

    def record_epoch(self, epoch: int, **extra_kwargs) -> None:
        """
        Promedia las métricas de todos los batches de la época y escribe
        una fila en el CSV de épocas.

        Parámetros adicionales (acc_natural, acc_robust, etc.) se ignoran
        aquí pero podrían extenderse en el futuro.
        """
        if not self._epoch_batches:
            return

        avg_row: dict = {"identifier": f"epoca_{epoch}"}

        # Promediar todas las columnas numéricas
        numeric_keys = [k for k in self._FIELDNAMES if k != "identifier"]
        for key in numeric_keys:
            vals = [b[key] for b in self._epoch_batches
                    if key in b and not _is_nan(b[key])]
            avg_row[key] = float(np.mean(vals)) if vals else float("nan")

        self._write_row(self._epoch_csv, avg_row)
        self._epoch_batches = []

        print(f"[METRICS] Métricas época {epoch} guardadas en: {self._results_dir}")

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _compute_row(self, identifier: str, info: dict) -> dict:
        """Calcula todas las estadísticas a partir del info dict de un batch."""
        lam         = info["lam"]
        entropy     = info["entropy"]
        sensitivity = info["sensitivity"]
        predictions = info["predictions"]
        targets     = info["targets"]

        # Estadísticas básicas
        lam_s    = _stats(lam)
        h_s      = _stats(entropy)
        s_s      = _stats(sensitivity)

        # Correlaciones de lambda con sus componentes
        corr_lam_sens   = _safe_corrcoef(lam, sensitivity)
        corr_lam_ent    = _safe_corrcoef(lam, entropy)

        # Métricas por tipo de muestra (correct vs incorrect)
        correct_mask   = (predictions == targets)
        incorrect_mask = ~correct_mask

        lam_correct   = float(np.mean(lam[correct_mask]))   if correct_mask.any()   else float("nan")
        lam_incorrect = float(np.mean(lam[incorrect_mask])) if incorrect_mask.any() else float("nan")

        row = {
            "identifier":           identifier,
            # Lambda final
            "lam_std":              lam_s["std"],
            "lam_mean":             lam_s["mean"],
            "lam_min":              lam_s["min"],
            "lam_max":              lam_s["max"],
            # Entropía
            "H_std":                h_s["std"],
            "H_mean":               h_s["mean"],
            "H_min":                h_s["min"],
            "H_max":                h_s["max"],
            # Sensibilidad
            "S_std":                s_s["std"],
            "S_mean":               s_s["mean"],
            "S_min":                s_s["min"],
            "S_max":                s_s["max"],
            # Correlaciones
            "corr_lam_sensitivity": corr_lam_sens,
            "corr_lam_entropy":     corr_lam_ent,
            # Correct/Incorrect
            "lam_correct_mean":     lam_correct,
            "lam_incorrect_mean":   lam_incorrect,
            # Losses
            "loss_natural":         info.get("loss_natural", float("nan")),
            "loss_robust":          info.get("loss_robust",  float("nan")),
        }
        return row

    def _write_row(self, path: str, row: dict) -> None:
        """Escribe una fila al CSV especificado (modo append)."""
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writerow(row)


def _is_nan(val) -> bool:
    """Comprueba si un valor es NaN (seguro para int/str)."""
    try:
        return math.isnan(val)
    except (TypeError, ValueError):
        return False