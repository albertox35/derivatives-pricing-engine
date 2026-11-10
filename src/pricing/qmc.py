"""Generacion cuasi-aleatoria para Monte Carlo.

Tres decisiones tecnicas gobiernan este modulo, y las tres se toman mal con
frecuencia.

**1. La secuencia de Sobol se toma de SciPy, no se reimplementa.**
``scipy.stats.qmc.Sobol`` implementa los numeros de direccion de Joe-Kuo,
validados hasta 21201 dimensiones. Reescribirla introduciria riesgo sin ganancia:
el costo del Monte Carlo esta en construir trayectorias y evaluar payoffs, no en
generar los puntos. Ese bucle si esta en C++.

**2. Sin mezclado (*scrambling*) no se puede estimar el error.**
Una secuencia de Sobol cruda es determinista: una sola corrida da un numero sin
intervalo de confianza. El mezclado de Owen aleatoriza la secuencia preservando
la propiedad de baja discrepancia, con lo que varias replicas independientes
permiten estimar el error. Reportar un precio de QMC crudo con un «error
estandar» calculado sobre las rutas es incorrecto: esas rutas no son
independientes.

**3. El numero de puntos debe ser potencia de dos.**
Las propiedades de equidistribucion de Sobol se cumplen en bloques de
:math:`2^m`. Pedir 10 000 puntos rompe el balance de la red y degrada la
convergencia. Aqui se exige la potencia de dos de forma explicita en lugar de
dejar que el usuario pierda precision sin enterarse.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, qmc

__all__ = ["next_power_of_two", "sobol_normals", "pseudorandom_normals", "replicated_sobol_normals"]

_CLIP = 1e-12


def next_power_of_two(n: int) -> int:
    """Menor potencia de dos mayor o igual que ``n``."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def sobol_normals(
    n_paths: int,
    n_dims: int,
    seed: int | None = 20260101,
    scramble: bool = True,
    strict_power_of_two: bool = True,
) -> np.ndarray:
    """Normales estandar a partir de una red de Sobol.

    La transformacion es por **inversa de la funcion de distribucion**, no por
    Box-Muller. Box-Muller consume dos dimensiones para producir dos normales
    mezclando sus roles, lo que destruye la estructura dimensional de la
    secuencia: justo lo que el puente browniano necesita preservar para que el
    ordenamiento por varianza funcione.

    Parameters
    ----------
    n_paths:
        Numero de rutas. Debe ser potencia de dos si
        ``strict_power_of_two`` es ``True``.
    n_dims:
        Dimension, igual al numero de pasos temporales.
    seed:
        Semilla del mezclado.
    scramble:
        Mezclado de Owen. Desactivarlo da la secuencia cruda, util solo para
        exhibir su estructura determinista.
    strict_power_of_two:
        Rechaza tamanios que no sean potencia de dos.

    Returns
    -------
    numpy.ndarray
        Matriz ``(n_paths, n_dims)`` contigua en C y de tipo ``float64``, lista
        para que el nucleo nativo la consuma sin copiar.
    """
    if n_paths <= 0 or n_dims <= 0:
        raise ValueError("n_paths y n_dims deben ser positivos")
    if strict_power_of_two and (n_paths & (n_paths - 1)) != 0:
        raise ValueError(
            f"n_paths = {n_paths} no es potencia de dos. Las propiedades de "
            f"equidistribucion de Sobol se cumplen en bloques de 2^m; usar "
            f"{next_power_of_two(n_paths)} o pasar strict_power_of_two=False "
            "aceptando la perdida de convergencia."
        )

    engine = qmc.Sobol(d=n_dims, scramble=scramble, seed=seed)
    points = engine.random(n_paths)
    # Sin mezclado el primer punto es exactamente cero, cuya inversa es -inf.
    np.clip(points, _CLIP, 1.0 - _CLIP, out=points)
    return np.ascontiguousarray(norm.ppf(points), dtype=np.float64)


def replicated_sobol_normals(
    n_paths: int,
    n_dims: int,
    n_replications: int = 16,
    seed: int = 20260101,
) -> list[np.ndarray]:
    """Replicas independientes de Sobol mezclado.

    Es la forma correcta de obtener un intervalo de confianza con QMC: cada
    replica usa un mezclado distinto, el estimador es el promedio de las
    replicas, y el error estandar se calcula **entre replicas**, que si son
    independientes.

    Returns
    -------
    list[numpy.ndarray]
        ``n_replications`` matrices ``(n_paths, n_dims)``.
    """
    if n_replications < 2:
        raise ValueError("se requieren al menos 2 replicas para estimar el error")
    rng = np.random.default_rng(seed)
    semillas = rng.integers(0, 2**31 - 1, size=n_replications)
    return [
        sobol_normals(n_paths, n_dims, seed=int(s), scramble=True) for s in semillas
    ]


def pseudorandom_normals(n_paths: int, n_dims: int, seed: int | None = 20260101) -> np.ndarray:
    """Normales pseudoaleatorias, para comparar contra QMC.

    Returns
    -------
    numpy.ndarray
        Matriz ``(n_paths, n_dims)`` contigua en C.
    """
    if n_paths <= 0 or n_dims <= 0:
        raise ValueError("n_paths y n_dims deben ser positivos")
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((n_paths, n_dims)), dtype=np.float64)
