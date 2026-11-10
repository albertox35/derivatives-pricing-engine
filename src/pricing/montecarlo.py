"""Monte Carlo y Monte Carlo cuasi-aleatorio sobre el nucleo nativo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from . import _kernel
from .qmc import pseudorandom_normals, replicated_sobol_normals, sobol_normals

__all__ = ["Payoff", "MonteCarloResult", "price_mc", "price_qmc", "price_qmc_replicated"]


class Payoff(IntEnum):
    """Payoffs soportados por el nucleo nativo.

    Los codigos coinciden con el ``enum class PayoffKind`` de ``cpp/kernel.cpp``;
    el ``IntEnum`` evita pasar enteros desnudos a traves de la frontera.
    """

    CALL = 0
    PUT = 1
    ASIAN_CALL = 2
    ASIAN_PUT = 3
    LOOKBACK_CALL = 4


@dataclass(frozen=True)
class MonteCarloResult:
    """Resultado de una valuacion por simulacion."""

    price: float
    standard_error: float
    n_paths: int
    n_steps: int
    method: str

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """Intervalo de confianza al 95 %."""
        half = 1.959963984540054 * self.standard_error
        return (self.price - half, self.price + half)

    def __str__(self) -> str:
        return (
            f"{self.method}: {self.price:.6f} +/- {self.standard_error:.6f} "
            f"({self.n_paths} rutas x {self.n_steps} pasos)"
        )


def _price_from_normals(
    normals: np.ndarray,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    payoff: Payoff,
    use_bridge: bool,
    n_threads: int,
) -> dict:
    return _kernel.price_gbm(
        normals,
        spot,
        strike,
        rate,
        dividend,
        vol,
        maturity,
        int(payoff),
        use_bridge,
        n_threads,
    )


def price_mc(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    payoff: Payoff = Payoff.CALL,
    n_paths: int = 100_000,
    n_steps: int = 32,
    seed: int | None = 20260101,
    use_bridge: bool = False,
    n_threads: int = 0,
) -> MonteCarloResult:
    """Valuacion por Monte Carlo pseudoaleatorio.

    Aqui el error estandar **si** es valido: las rutas son independientes, asi
    que la varianza muestral estima correctamente el error del promedio.
    """
    normals = pseudorandom_normals(n_paths, n_steps, seed)
    out = _price_from_normals(
        normals, spot, strike, rate, dividend, vol, maturity, payoff, use_bridge, n_threads
    )
    return MonteCarloResult(
        price=float(out["price"]),
        standard_error=float(out["standard_error"]),
        n_paths=int(out["n_paths"]),
        n_steps=int(out["n_steps"]),
        method="MC pseudoaleatorio",
    )


def price_qmc(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    payoff: Payoff = Payoff.CALL,
    n_paths: int = 65_536,
    n_steps: int = 32,
    seed: int | None = 20260101,
    use_bridge: bool = True,
    n_threads: int = 0,
) -> MonteCarloResult:
    """Valuacion por Monte Carlo cuasi-aleatorio con puente browniano.

    Warning
    -------
    El ``standard_error`` que devuelve **no es un error estandar valido**. Los
    puntos de una red de Sobol no son independientes, asi que la varianza
    muestral sobre las rutas no estima el error del estimador. Se devuelve solo
    como medida de dispersion del payoff. Para un intervalo de confianza
    legitimo hay que usar :func:`price_qmc_replicated`.
    """
    normals = sobol_normals(n_paths, n_steps, seed=seed, scramble=True)
    out = _price_from_normals(
        normals, spot, strike, rate, dividend, vol, maturity, payoff, use_bridge, n_threads
    )
    return MonteCarloResult(
        price=float(out["price"]),
        standard_error=float(out["standard_error"]),
        n_paths=int(out["n_paths"]),
        n_steps=int(out["n_steps"]),
        method="QMC Sobol + puente browniano",
    )


def price_qmc_replicated(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    payoff: Payoff = Payoff.CALL,
    n_paths: int = 8192,
    n_steps: int = 32,
    n_replications: int = 16,
    seed: int = 20260101,
    use_bridge: bool = True,
    n_threads: int = 0,
) -> MonteCarloResult:
    """QMC con replicas mezcladas, que si da un error estandar valido.

    Cada replica usa un mezclado de Owen distinto. El estimador es el promedio de
    las replicas y el error estandar se calcula **entre** ellas, que son
    independientes por construccion.

    Returns
    -------
    MonteCarloResult
        ``n_paths`` reporta el total agregado sobre las replicas.
    """
    replicas = replicated_sobol_normals(n_paths, n_steps, n_replications, seed)
    estimaciones = np.array(
        [
            float(
                _price_from_normals(
                    z, spot, strike, rate, dividend, vol, maturity, payoff, use_bridge, n_threads
                )["price"]
            )
            for z in replicas
        ]
    )
    media = float(np.mean(estimaciones))
    se = float(np.std(estimaciones, ddof=1) / np.sqrt(n_replications))
    return MonteCarloResult(
        price=media,
        standard_error=se,
        n_paths=n_paths * n_replications,
        n_steps=n_steps,
        method=f"QMC replicado ({n_replications} mezclados)",
    )
