"""Motor de valuacion de derivados con nucleo nativo C++20.

Tres rutas de valuacion, todas contrastadas contra Black-Scholes en forma
cerrada:

- **Monte Carlo cuasi-aleatorio** con red de Sobol mezclada y puente browniano
  con ordenamiento por varianza.
- **Diferencias finitas** por Crank-Nicolson con arranque de Rannacher, para
  europeas y americanas.
- **Diferenciacion automatica en modo reverso**, que entrega todas las griegas en
  un solo recorrido.

El nucleo nativo recibe los arreglos de NumPy sin copiarlos y libera el GIL
durante el computo.
"""

from __future__ import annotations

from . import black_scholes
from .greeks import SimulatedGreeks, aad_greeks, likelihood_ratio_greeks, pathwise_greeks
from .montecarlo import (
    MonteCarloResult,
    Payoff,
    price_mc,
    price_qmc,
    price_qmc_replicated,
)
from .pde import PDEResult, gamma_curve, solve_american, solve_european
from .qmc import pseudorandom_normals, replicated_sobol_normals, sobol_normals

__version__ = "0.1.0"


def has_openmp() -> bool:
    """``True`` si el nucleo se compilo con OpenMP."""
    from . import _kernel

    return bool(_kernel.__has_openmp__)


__all__ = [
    "MonteCarloResult",
    "PDEResult",
    "Payoff",
    "SimulatedGreeks",
    "__version__",
    "aad_greeks",
    "black_scholes",
    "gamma_curve",
    "has_openmp",
    "likelihood_ratio_greeks",
    "pathwise_greeks",
    "price_mc",
    "price_qmc",
    "price_qmc_replicated",
    "pseudorandom_normals",
    "replicated_sobol_normals",
    "sobol_normals",
    "solve_american",
    "solve_european",
]
