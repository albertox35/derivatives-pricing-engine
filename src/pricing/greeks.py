"""Estimadores de sensibilidades por simulacion.

Hay tres familias, y la eleccion entre ellas no es de gusto: cada una falla en
casos concretos y hay que saber cuales.

**Diferencias finitas.** Reevalua el precio con el parametro perturbado. Cuesta
una revaluacion completa por parametro, y el sesgo y la varianza compiten: un
paso grande sesga, uno pequenio hace que el ruido de Monte Carlo domine. Es el
metodo de referencia de ultimo recurso, no el de produccion.

**Pathwise (derivada de la trayectoria).** Deriva el payoff respecto al
parametro dentro de la esperanza. Tiene varianza baja y no tiene sesgo, pero
**exige que el payoff sea diferenciable casi seguramente**. Funciona para delta
y vega de un call, y **falla para gamma**: la derivada del payoff de un call es
una indicadora, cuya derivada es una delta de Dirac que el estimador no puede
representar. Esa falla se demuestra en las pruebas, no se menciona de pasada.

**Razon de verosimilitud (*likelihood ratio*).** Deriva la densidad en lugar del
payoff. Admite payoffs discontinuos y resuelve gamma, a cambio de mas varianza.

**AD en modo reverso** (en el nucleo nativo, :func:`pricing.aad_greeks`) da
todas las sensibilidades en un solo recorrido, a un costo independiente del
numero de parametros. Es lo que hace viable calcular sensibilidades de un
portafolio con cientos de factores, que es el caso de XVA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _kernel

__all__ = ["SimulatedGreeks", "pathwise_greeks", "likelihood_ratio_greeks", "aad_greeks"]


@dataclass(frozen=True)
class SimulatedGreeks:
    """Sensibilidades estimadas por simulacion, con su error estandar."""

    delta: float
    delta_stderr: float
    vega: float | None = None
    vega_stderr: float | None = None
    gamma: float | None = None
    gamma_stderr: float | None = None
    method: str = ""


def _terminal_values(
    spot: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    n_paths: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Valores terminales del GBM y las normales que los generaron."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    drift = (rate - dividend - 0.5 * vol**2) * maturity
    s_t = spot * np.exp(drift + vol * np.sqrt(maturity) * z)
    return s_t, z


def pathwise_greeks(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    n_paths: int = 500_000,
    seed: int | None = 20260101,
) -> SimulatedGreeks:
    """Delta y vega de un call europeo por derivada de la trayectoria.

    .. math::

        \\Delta = E\\left[e^{-rT}\\,\\mathbf{1}_{\\{S_T>K\\}}\\,\\frac{S_T}{S_0}\\right]

    .. math::

        \\mathcal{V} = E\\left[e^{-rT}\\,\\mathbf{1}_{\\{S_T>K\\}}\\,S_T
            \\left(\\frac{\\ln(S_T/S_0) - (r-q-\\sigma^2/2)T}{\\sigma}
                  - \\sigma T\\right)\\right]

    No se devuelve gamma: el estimador pathwise **no existe** para la gamma de un
    call, porque requeriria derivar una funcion indicadora.
    """
    s_t, _ = _terminal_values(spot, rate, dividend, vol, maturity, n_paths, seed)
    df = np.exp(-rate * maturity)
    dentro = s_t > strike

    delta_muestra = df * dentro * s_t / spot

    # dS_T/dsigma = S_T (sqrt(T) Z - sigma T), con sqrt(T)Z despejado del
    # propio S_T para no arrastrar Z por separado.
    sqrt_t_z = (np.log(s_t / spot) - (rate - dividend - 0.5 * vol**2) * maturity) / vol
    vega_muestra = df * dentro * s_t * (sqrt_t_z - vol * maturity)

    n = float(n_paths)
    return SimulatedGreeks(
        delta=float(np.mean(delta_muestra)),
        delta_stderr=float(np.std(delta_muestra, ddof=1) / np.sqrt(n)),
        vega=float(np.mean(vega_muestra)),
        vega_stderr=float(np.std(vega_muestra, ddof=1) / np.sqrt(n)),
        gamma=None,
        gamma_stderr=None,
        method="pathwise",
    )


def likelihood_ratio_greeks(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    n_paths: int = 500_000,
    seed: int | None = 20260101,
    is_call: bool = True,
) -> SimulatedGreeks:
    """Delta y gamma por razon de verosimilitud.

    Se deriva la densidad lognormal respecto al spot en lugar del payoff, con lo
    que el payoff puede ser discontinuo:

    .. math::

        \\Delta = E\\left[e^{-rT} g(S_T)\\, \\frac{Z}{S_0\\sigma\\sqrt{T}}\\right]

    .. math::

        \\Gamma = E\\left[e^{-rT} g(S_T)
            \\left(\\frac{Z^2-1}{S_0^2\\sigma^2 T}
                 - \\frac{Z}{S_0^2\\sigma\\sqrt{T}}\\right)\\right]

    Esta es la via practica para la gamma de un payoff con kink, donde el
    estimador pathwise no esta definido.
    """
    s_t, z = _terminal_values(spot, rate, dividend, vol, maturity, n_paths, seed)
    df = np.exp(-rate * maturity)
    g = np.maximum(s_t - strike, 0.0) if is_call else np.maximum(strike - s_t, 0.0)
    sqrt_t = np.sqrt(maturity)

    score_delta = z / (spot * vol * sqrt_t)
    score_gamma = (z**2 - 1.0) / (spot**2 * vol**2 * maturity) - z / (spot**2 * vol * sqrt_t)

    delta_muestra = df * g * score_delta
    gamma_muestra = df * g * score_gamma

    n = float(n_paths)
    return SimulatedGreeks(
        delta=float(np.mean(delta_muestra)),
        delta_stderr=float(np.std(delta_muestra, ddof=1) / np.sqrt(n)),
        gamma=float(np.mean(gamma_muestra)),
        gamma_stderr=float(np.std(gamma_muestra, ddof=1) / np.sqrt(n)),
        method="razon de verosimilitud",
    )


def aad_greeks(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    is_call: bool = True,
) -> dict[str, float]:
    """Todas las griegas por AD en modo reverso, en un solo recorrido.

    Envoltura del nucleo nativo. Devuelve precio, delta, vega, rho, theta, la
    sensibilidad al dividendo y la **dual delta** (derivada respecto al strike,
    que es la que se necesita para extraer densidades implicitas del mercado).

    El costo es independiente del numero de parametros: la cinta se recorre una
    vez hacia atras y entrega todas las derivadas. Con diferencias finitas harian
    falta seis revaluaciones completas.

    Returns
    -------
    dict[str, float]
        Incluye ``tape_size``, el numero de nodos de la cinta, como medida del
        costo del recorrido reverso.
    """
    out = _kernel.black_scholes_aad(spot, strike, rate, dividend, vol, maturity, is_call)
    return {k: float(v) for k, v in out.items()}
