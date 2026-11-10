"""Black-Scholes en forma cerrada.

Existe por una sola razon: ser la **referencia de validacion** del resto del
motor. El Monte Carlo, el solver PDE y la cinta de AD se contrastan contra estas
formulas, que son exactas.

Un motor numerico sin un caso con solucion cerrada contra el que compararse no es
verificable, y ninguna cifra que produzca es defendible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = ["BlackScholesGreeks", "price", "greeks", "implied_volatility"]


@dataclass(frozen=True)
class BlackScholesGreeks:
    """Precio y sensibilidades analiticas."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        """Representacion como diccionario, para comparar con otros motores."""
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


def _d1_d2(
    spot: float, strike: float, rate: float, dividend: float, vol: float, maturity: float
) -> tuple[float, float]:
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * vol**2) * maturity) / (vol * sqrt_t)
    return float(d1), float(d1 - vol * sqrt_t)


def _validate(spot: float, strike: float, vol: float, maturity: float) -> None:
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot y strike deben ser positivos")
    if vol <= 0.0:
        raise ValueError("la volatilidad debe ser positiva")
    if maturity <= 0.0:
        raise ValueError("el vencimiento debe ser positivo")


def price(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    is_call: bool = True,
) -> float:
    """Precio de una opcion europea.

    Examples
    --------
    >>> round(price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0), 6)
    9.227006
    """
    _validate(spot, strike, vol, maturity)
    d1, d2 = _d1_d2(spot, strike, rate, dividend, vol, maturity)
    df_r = np.exp(-rate * maturity)
    df_q = np.exp(-dividend * maturity)
    if is_call:
        return float(spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2))
    return float(strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1))


def greeks(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    is_call: bool = True,
) -> BlackScholesGreeks:
    """Precio y las cinco griegas de primer y segundo orden.

    ``theta`` se define como la derivada respecto al **paso del tiempo**, es
    decir el negativo de la derivada respecto al vencimiento restante. Es la
    convencion de mesa, y la misma que usa la cinta de AD del nucleo nativo para
    que las dos sean comparables.
    """
    _validate(spot, strike, vol, maturity)
    d1, d2 = _d1_d2(spot, strike, rate, dividend, vol, maturity)
    sqrt_t = np.sqrt(maturity)
    df_r = np.exp(-rate * maturity)
    df_q = np.exp(-dividend * maturity)
    pdf_d1 = float(norm.pdf(d1))

    gamma = df_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * df_q * pdf_d1 * sqrt_t

    if is_call:
        valor = spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2)
        delta = df_q * float(norm.cdf(d1))
        rho = strike * maturity * df_r * float(norm.cdf(d2))
        theta = (
            -spot * df_q * pdf_d1 * vol / (2.0 * sqrt_t)
            + dividend * spot * df_q * float(norm.cdf(d1))
            - rate * strike * df_r * float(norm.cdf(d2))
        )
    else:
        valor = strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1)
        delta = -df_q * float(norm.cdf(-d1))
        rho = -strike * maturity * df_r * float(norm.cdf(-d2))
        theta = (
            -spot * df_q * pdf_d1 * vol / (2.0 * sqrt_t)
            - dividend * spot * df_q * float(norm.cdf(-d1))
            + rate * strike * df_r * float(norm.cdf(-d2))
        )

    return BlackScholesGreeks(
        price=float(valor),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def implied_volatility(
    target_price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    maturity: float,
    is_call: bool = True,
    tol: float = 1e-12,
    max_iter: int = 100,
    min_vega: float = 1e-6,
) -> float:
    """Volatilidad implicita por Newton con respaldo de biseccion.

    Newton converge rapido porque vega tiene forma cerrada, pero **no es seguro
    por si solo**: para opciones muy dentro o muy fuera del dinero vega tiende a
    cero y el paso de Newton explota. Por eso se mantiene un intervalo de
    biseccion y se rechaza cualquier iterado que salga de el.

    Hay un problema mas profundo que la estabilidad del algoritmo: cuando vega es
    practicamente cero, la volatilidad implicita **no es identificable**. Muchas
    sigmas reproducen el mismo precio dentro de la tolerancia, asi que no hay una
    respuesta que devolver. La funcion lo detecta y lanza una excepcion en lugar
    de devolver el primer valor que converja, porque ese numero no significaria
    nada.

    Conviene no confundir dos cosas que suelen mezclarse:

    - **Vega depende del strike, no del instrumento.** Por paridad put-call, el
      call y el put del mismo strike y plazo tienen *exactamente* la misma vega.
      Si vega es despreciable lo es para los dos, y cambiar de instrumento no
      resuelve nada: hace falta un strike mas cercano al dinero o un plazo mas
      largo.
    - **Las mesas cotizan el lado fuera del dinero por acondicionamiento del
      precio, no porque tenga mas vega.** En una opcion muy dentro del dinero el
      valor temporal es una fraccion minuscula del precio, asi que el redondeo de
      la cotizacion se traduce en un error grande de volatilidad. El instrumento
      fuera del dinero tiene la misma vega sobre un precio mucho menor, de modo
      que la informacion sobre volatilidad pesa mucho mas en lo que se cotiza.

    Parameters
    ----------
    min_vega:
        Vega minima en la solucion para considerar identificable el problema.

    Raises
    ------
    ValueError
        Si el precio objetivo viola las cotas de no-arbitraje, o si la
        volatilidad implicita no es identificable por vega insuficiente.
    """
    _validate(spot, strike, 1.0, maturity)
    df_r = np.exp(-rate * maturity)
    df_q = np.exp(-dividend * maturity)

    if is_call:
        lower_bound = max(spot * df_q - strike * df_r, 0.0)
        upper_bound = spot * df_q
    else:
        lower_bound = max(strike * df_r - spot * df_q, 0.0)
        upper_bound = strike * df_r

    if not (lower_bound - 1e-12 <= target_price <= upper_bound + 1e-12):
        raise ValueError(
            f"el precio {target_price} esta fuera de las cotas de no-arbitraje "
            f"[{lower_bound:.6f}, {upper_bound:.6f}]: no existe volatilidad implicita"
        )

    def check_identifiable(sigma: float) -> float:
        v = greeks(spot, strike, rate, dividend, sigma, maturity, is_call).vega
        if v >= min_vega:
            return sigma
        desviaciones = abs(float(np.log(strike / spot))) / (max(sigma, 1e-12) * np.sqrt(maturity))
        raise ValueError(
            f"la volatilidad implicita no es identificable: vega = {v:.3e} en "
            f"sigma = {sigma:.4f}, por debajo de min_vega = {min_vega:.1e}. El strike "
            f"esta a {desviaciones:.1f} desviaciones estandar del spot para esta "
            "volatilidad y plazo, asi que el precio es casi insensible a la "
            "volatilidad y muchas sigmas lo reproducen. Cambiar de call a put NO "
            "ayuda: por paridad put-call vega es identica para los dos instrumentos "
            "del mismo strike. Hace falta un strike mas cercano al dinero, o un plazo "
            "mas largo."
        )

    lo, hi = 1e-9, 10.0
    sigma = 0.2
    for _ in range(max_iter):
        valor = price(spot, strike, rate, dividend, sigma, maturity, is_call)
        diff = valor - target_price
        if abs(diff) < tol:
            return float(check_identifiable(sigma))
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        v = greeks(spot, strike, rate, dividend, sigma, maturity, is_call).vega
        if v > 1e-12:
            candidato = sigma - diff / v
            if lo < candidato < hi:
                sigma = candidato
                continue
        sigma = 0.5 * (lo + hi)
    return float(check_identifiable(sigma))
