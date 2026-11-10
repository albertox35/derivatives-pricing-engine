"""Solver de diferencias finitas: Crank-Nicolson con arranque de Rannacher.

El detalle que decide si un solver de este tipo sirve o no es el **arranque de
Rannacher**, y es el que suele omitirse.

Crank-Nicolson es de segundo orden en tiempo, pero no es L-estable: su factor de
amplificacion tiende a ``-1`` para los modos de alta frecuencia, en lugar de a
cero. Una condicion inicial con derivada discontinua --el kink del payoff en el
strike-- excita justamente esos modos, que entonces no se amortiguan sino que
oscilan de paso en paso.

El efecto en el precio es pequenio y facil de pasar por alto. En **gamma**, que
es una segunda derivada, es grande y visible: aparecen oscilaciones espurias
alrededor del strike.

El arranque de Rannacher sustituye los primeros pasos por medios pasos
totalmente implicitos, que si son L-estables y amortiguan esos modos, sin
degradar el orden de convergencia global. Es barato y resuelve el problema por
completo; el modulo lo activa por omision y permite desactivarlo justo para poder
**demostrar** el deterioro, que es lo que hace la bateria de pruebas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _kernel

__all__ = ["PDEResult", "solve_european", "solve_american", "gamma_curve"]


@dataclass(frozen=True)
class PDEResult:
    """Resultado del solver PDE."""

    price: float
    delta: float
    gamma: float
    n_space: int
    n_time: int
    rannacher_steps: int
    spot_grid: np.ndarray | None = None
    value_grid: np.ndarray | None = None

    def as_dict(self) -> dict[str, float]:
        """Representacion como diccionario, solo con los escalares."""
        return {"price": self.price, "delta": self.delta, "gamma": self.gamma}

    def gamma_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Gamma en todos los nodos interiores de la malla.

        Es la vista que revela la oscilacion de Crank-Nicolson: en un solo punto
        el precio puede parecer correcto mientras la segunda derivada oscila a lo
        largo de la malla. Sin esta vista la ventaja de Rannacher no es
        verificable.

        Returns
        -------
        spots, gammas
            Nodos interiores y la gamma estimada por diferencias de segundo orden
            sobre malla no uniforme en ``S``.
        """
        if self.spot_grid is None or self.value_grid is None:
            raise ValueError("volver a resolver con return_grid=True para obtener el perfil")
        s = self.spot_grid
        v = self.value_grid
        h1 = s[1:-1] - s[:-2]
        h2 = s[2:] - s[1:-1]
        gammas = 2.0 * (h1 * v[2:] - (h1 + h2) * v[1:-1] + h2 * v[:-2]) / (h1 * h2 * (h1 + h2))
        return s[1:-1], gammas


def gamma_curve(result: PDEResult, s_min: float, s_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Perfil de gamma restringido a un rango de spot."""
    spots, gammas = result.gamma_profile()
    mask = (spots >= s_min) & (spots <= s_max)
    return spots[mask], gammas[mask]


def _solve(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    n_space: int,
    n_time: int,
    is_call: bool,
    american: bool,
    rannacher_steps: int,
    n_stddev: float,
    return_grid: bool = False,
) -> PDEResult:
    if n_space % 2 != 0:
        # El nodo central debe caer exactamente en log(spot) para leer el precio
        # y las griegas sin interpolar.
        raise ValueError("n_space debe ser par para que el spot caiga en un nodo de la malla")
    out = _kernel.solve_pde(
        spot,
        strike,
        rate,
        dividend,
        vol,
        maturity,
        n_space,
        n_time,
        is_call,
        american,
        rannacher_steps,
        n_stddev,
        return_grid,
    )
    return PDEResult(
        price=float(out["price"]),
        delta=float(out["delta"]),
        gamma=float(out["gamma"]),
        n_space=n_space,
        n_time=n_time,
        rannacher_steps=rannacher_steps,
        spot_grid=out.get("spot_grid"),
        value_grid=out.get("value_grid"),
    )


def solve_european(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    n_space: int = 400,
    n_time: int = 200,
    is_call: bool = True,
    rannacher_steps: int = 2,
    n_stddev: float = 6.0,
    return_grid: bool = False,
) -> PDEResult:
    """Valua una opcion europea por diferencias finitas.

    Parameters
    ----------
    n_space:
        Nodos espaciales; debe ser par.
    n_time:
        Pasos temporales.
    rannacher_steps:
        Pasos iniciales que se parten en medios pasos implicitos. Con ``0`` se
        obtiene Crank-Nicolson puro, cuya gamma oscila cerca del strike.
    n_stddev:
        Semiancho del dominio en desviaciones estandar de ``log S``. Seis es
        holgado: la probabilidad de alcanzar la frontera es despreciable y el
        error de truncamiento del dominio queda por debajo del de discretizacion.

    Returns
    -------
    PDEResult
    """
    return _solve(
        spot,
        strike,
        rate,
        dividend,
        vol,
        maturity,
        n_space,
        n_time,
        is_call,
        False,
        rannacher_steps,
        n_stddev,
        return_grid,
    )


def solve_american(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    maturity: float,
    n_space: int = 400,
    n_time: int = 400,
    is_call: bool = False,
    rannacher_steps: int = 2,
    n_stddev: float = 6.0,
    return_grid: bool = False,
) -> PDEResult:
    """Valua una opcion americana por proyeccion sobre la restriccion de ejercicio.

    Tras cada paso temporal se aplica ``v = max(v, payoff)``. Es el metodo de
    proyeccion explicita, suficientemente preciso para una malla fina y mucho mas
    simple que resolver el problema de complementariedad lineal por PSOR.

    El caso que importa es el **put americano**: con tasa positiva su valor excede
    estrictamente al europeo, porque ejercer anticipadamente permite cobrar el
    strike e invertirlo. Un call americano sobre un subyacente sin dividendos, en
    cambio, nunca conviene ejercerlo antes, y su valor coincide con el europeo.
    Las dos propiedades se verifican en las pruebas.

    Returns
    -------
    PDEResult
    """
    return _solve(
        spot,
        strike,
        rate,
        dividend,
        vol,
        maturity,
        n_space,
        n_time,
        is_call,
        True,
        rannacher_steps,
        n_stddev,
        return_grid,
    )
