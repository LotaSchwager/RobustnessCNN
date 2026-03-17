"""
Metaheurísticas para optimización adaptativa de alpha y beta en D-TRADES.

Interfaz de uso durante el entrenamiento:
    mh = LocalSearchMetaheuristic(alpha0=1.0, beta0=1.0)
    for epoch in range(epochs):
        alpha, beta = mh.alpha, mh.beta
        loss = ...  # entrenar con alpha, beta actuales
        mh.update(loss)  # propone nuevo (alpha, beta) para la siguiente época

Para añadir una nueva metaheurística (Puma Optimizer, Grey Wolf, ACO, etc.):
    1. Heredar de BaseMetaheuristic.
    2. Implementar update(loss) -> (float, float).
    3. Registrar la clase en __init__.py.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


# =============================================================================
# Clase base (interfaz genérica)
# =============================================================================

class BaseMetaheuristic(ABC):
    """
    Interfaz estándar para metaheurísticas que optimizan alpha y beta.

    La función objetivo es la pérdida del modelo (se MINIMIZA).
    Cada época de entrenamiento cuenta como una evaluación de la función objetivo.

    Subclases obligatorias:
        update(loss) -> (alpha, beta)
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        self._alpha = float(alpha0)
        self._beta  = float(beta0)

    @abstractmethod
    def update(self, loss: float) -> Tuple[float, float]:
        """
        Recibe la pérdida de la época actual y devuelve el siguiente
        par (alpha, beta) para la próxima época.

        Parámetros
        ----------
        loss : float
            Pérdida total del modelo en la época que acaba de terminar.

        Retorna
        -------
        (alpha, beta) : Tuple[float, float]
            Nuevos valores a usar en la siguiente época.
        """
        ...

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def best_alpha(self) -> float:
        """Mejor alpha encontrado hasta el momento."""
        return self._alpha

    @property
    def best_beta(self) -> float:
        """Mejor beta encontrado hasta el momento."""
        return self._beta

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"alpha={self._alpha:.4f}, beta={self._beta:.4f})"
        )


# =============================================================================
# Implementación: Búsqueda Local por Cuadrícula
# =============================================================================

class LocalSearchMetaheuristic(BaseMetaheuristic):
    """
    Búsqueda local por cuadrícula con refinamiento adaptativo.

    Opera en tres fases secuenciales, consumiendo una época de entrenamiento
    por evaluación (sin reentrenar el modelo completo):

    Fase 1 - GRID (exploración):
        Evalúa sistemáticamente una cuadrícula de candidatos (alpha, beta).
        El primer candidato es siempre (alpha0, beta0).

    Fase 2 - REFINE (refinamiento local):
        Una vez registrado el mejor de la cuadrícula, explora vecinos
        cercanos con paso más fino alrededor de ese punto.

    Fase 3 - EXPLOIT (explotación):
        Fija definitivamente el mejor par (alpha, beta) encontrado.

    Parámetros
    ----------
    alpha0, beta0       : valores iniciales (primer candidato evaluado).
    grid_values         : valores que toma cada parámetro en la cuadrícula.
                          Por defecto [0.0, 0.5, 1.0, 1.5, 2.0].
    refine_radius       : radio de búsqueda local tras la cuadrícula.
    refine_step         : paso del refinamiento local.
    lo, hi              : límites del espacio de búsqueda.
    verbose             : imprime cambios de fase y mejoras encontradas.
    """

    def __init__(
        self,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        grid_values: Optional[List[float]] = None,
        refine_radius: float = 0.25,
        refine_step: float = 0.125,
        lo: float = 0.0,
        hi: float = 3.0,
        verbose: bool = True,
    ):
        super().__init__(alpha0, beta0)

        self._lo           = lo
        self._hi           = hi
        self._verbose      = verbose
        self._refine_radius = refine_radius
        self._refine_step   = refine_step

        # Registro del mejor encontrado
        self._best_alpha = float(alpha0)
        self._best_beta  = float(beta0)
        self._best_loss  = float("inf")

        # ---- Fase 1: cuadrícula ----
        if grid_values is None:
            grid_values = [0.0, 0.5, 1.0, 1.5, 2.0]

        candidates: List[Tuple[float, float]] = []
        first = (float(alpha0), float(beta0))
        # (alpha0, beta0) siempre el primero en ser evaluado
        candidates.append(first)
        for a in grid_values:
            for b in grid_values:
                pair = (float(a), float(b))
                if pair != first:
                    candidates.append(pair)

        self._candidates: List[Tuple[float, float]] = candidates
        self._idx:   int = 0
        self._phase: str = "grid"

        # El primer candidato ya están en self._alpha / self._beta desde super().__init__

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def update(self, loss: float) -> Tuple[float, float]:
        """
        Registra la pérdida del candidato actual y avanza al siguiente.

        Retorna (alpha, beta) para la próxima época.
        """
        improved = loss < self._best_loss
        if self._verbose:
            print(
                f"[META][{self._phase.upper():7s}] "
                f"alpha={self._alpha:.4f}  beta={self._beta:.4f}  "
                f"loss={loss:.6f}"
                + ("  ← mejor" if improved else "")
            )

        # Actualiza mejor visto hasta ahora
        if improved:
            self._best_loss  = loss
            self._best_alpha = self._alpha
            self._best_beta  = self._beta

        # Avanza según la fase
        self._idx += 1

        if self._phase == "grid":
            if self._idx < len(self._candidates):
                self._alpha, self._beta = self._candidates[self._idx]
            else:
                self._start_refine()

        elif self._phase == "refine":
            if self._idx < len(self._candidates):
                self._alpha, self._beta = self._candidates[self._idx]
            else:
                self._start_exploit()

        # En "exploit" no hay más cambios de (alpha, beta)

        return self._alpha, self._beta

    # ------------------------------------------------------------------
    # Propiedades adicionales
    # ------------------------------------------------------------------

    @property
    def best_alpha(self) -> float:
        return self._best_alpha

    @property
    def best_beta(self) -> float:
        return self._best_beta

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def phase(self) -> str:
        return self._phase

    # ------------------------------------------------------------------
    # Transiciones de fase (privadas)
    # ------------------------------------------------------------------

    def _start_refine(self) -> None:
        if self._verbose:
            print(
                f"[META] ── Inicio REFINE alrededor de "
                f"alpha={self._best_alpha:.4f}, beta={self._best_beta:.4f} "
                f"(mejor loss={self._best_loss:.6f})"
            )

        # Genera vecinos en un entorno del mejor
        alphas = np.arange(
            self._best_alpha - self._refine_radius,
            self._best_alpha + self._refine_radius + 1e-9,
            self._refine_step,
        )
        betas = np.arange(
            self._best_beta - self._refine_radius,
            self._best_beta + self._refine_radius + 1e-9,
            self._refine_step,
        )

        candidates: set[Tuple[float, float]] = set()
        for a in alphas:
            a_c = round(float(np.clip(a, self._lo, self._hi)), 4)
            for b in betas:
                b_c = round(float(np.clip(b, self._lo, self._hi)), 4)
                candidates.add((a_c, b_c))

        # Excluye el punto del mejor ya evaluado
        candidates.discard(
            (round(self._best_alpha, 4), round(self._best_beta, 4))
        )

        self._candidates = sorted(candidates)
        self._phase = "refine"
        self._idx   = 0

        if self._candidates:
            self._alpha, self._beta = self._candidates[0]
        else:
            # No hay vecinos nuevos → explotación directa
            self._start_exploit()

    def _start_exploit(self) -> None:
        if self._verbose:
            print(
                f"[META] ── Inicio EXPLOIT → fijando "
                f"alpha={self._best_alpha:.4f}, beta={self._best_beta:.4f}"
            )
        self._phase = "exploit"
        self._alpha = self._best_alpha
        self._beta  = self._best_beta