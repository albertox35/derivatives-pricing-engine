"""Validacion de la simulacion: QMC, puente browniano y estimadores de griegas."""

from __future__ import annotations

import numpy as np
import pytest

from pricing import (
    Payoff,
    black_scholes as bs,
    likelihood_ratio_greeks,
    pathwise_greeks,
    price_mc,
    price_qmc,
    price_qmc_replicated,
    pseudorandom_normals,
    sobol_normals,
)
from pricing.qmc import next_power_of_two

S, K, R, Q, V, T = 100.0, 100.0, 0.05, 0.02, 0.20, 1.0
REF = bs.price(S, K, R, Q, V, T)


# ------------------------------------------------------------------- Sobol


def test_potencia_de_dos():
    assert next_power_of_two(1) == 1
    assert next_power_of_two(1000) == 1024
    assert next_power_of_two(1024) == 1024


def test_sobol_exige_potencia_de_dos():
    with pytest.raises(ValueError, match="no es potencia de dos"):
        sobol_normals(1000, 8)


@pytest.mark.filterwarnings("ignore:The balance properties of Sobol:UserWarning")
def test_sobol_permite_salirse_de_la_potencia_de_dos_de_forma_explicita():
    # SciPy avisa de la perdida de balance; es justo lo que se esta aceptando.
    z = sobol_normals(1000, 4, strict_power_of_two=False)
    assert z.shape == (1000, 4)


def test_las_normales_de_sobol_tienen_los_momentos_correctos():
    z = sobol_normals(65_536, 4, seed=1)
    assert abs(float(np.mean(z))) < 0.01
    assert float(np.std(z)) == pytest.approx(1.0, abs=0.01)


def test_el_mezclado_cambia_la_red_pero_no_su_distribucion():
    crudo = sobol_normals(4096, 3, scramble=False)
    mezclado_a = sobol_normals(4096, 3, seed=1, scramble=True)
    mezclado_b = sobol_normals(4096, 3, seed=2, scramble=True)
    # Dos mezclados distintos dan redes distintas: eso es lo que permite
    # estimar el error con replicas independientes.
    assert not np.allclose(mezclado_a, mezclado_b)
    assert not np.allclose(crudo, mezclado_a)
    # Pero la distribucion marginal se conserva.
    for z in (crudo, mezclado_a, mezclado_b):
        assert abs(float(np.mean(z))) < 0.05


def test_las_normales_salen_contiguas_y_en_float64():
    """Condicion para que el nucleo nativo las consuma sin copiar."""
    for z in (sobol_normals(1024, 4), pseudorandom_normals(1024, 4)):
        assert z.dtype == np.float64
        assert z.flags["C_CONTIGUOUS"]


# ---------------------------------------------------- convergencia MC vs QMC


def test_el_qmc_es_mucho_mas_preciso_que_el_mc_con_las_mismas_rutas():
    """El motivo de usar Sobol, medido en lugar de afirmado."""
    n = 4096
    err_mc = np.mean(
        [abs(price_mc(S, K, R, Q, V, T, Payoff.CALL, n, 16, seed=1000 + s).price - REF) for s in range(6)]
    )
    err_qmc = np.mean(
        [abs(price_qmc(S, K, R, Q, V, T, Payoff.CALL, n, 16, seed=1000 + s).price - REF) for s in range(6)]
    )
    assert err_qmc < err_mc / 20.0, f"MC {err_mc:.5f} vs QMC {err_qmc:.5f}"


def test_el_error_del_qmc_decrece_al_aumentar_las_rutas():
    errores = [
        np.mean(
            [
                abs(price_qmc(S, K, R, Q, V, T, Payoff.CALL, n, 16, seed=7 + s).price - REF)
                for s in range(4)
            ]
        )
        for n in (1024, 4096, 16_384)
    ]
    assert all(b < a for a, b in zip(errores, errores[1:])), errores


def test_el_puente_browniano_reduce_la_varianza_en_un_payoff_dependiente_de_la_trayectoria():
    """El puente solo sirve si reordena las dimensiones por varianza.

    En una opcion asiatica con 32 pasos, construir la trayectoria de forma
    incremental gasta las dimensiones de Sobol de indice bajo --las mejor
    equidistribuidas-- en incrementos de varianza pequenia. El puente las asigna
    primero al punto terminal y luego a los puntos medios, que es el orden
    correcto.
    """
    n = 8192
    con = [
        price_qmc(S, K, R, Q, V, T, Payoff.ASIAN_CALL, n, 32, seed=200 + s, use_bridge=True).price
        for s in range(6)
    ]
    sin = [
        price_qmc(S, K, R, Q, V, T, Payoff.ASIAN_CALL, n, 32, seed=200 + s, use_bridge=False).price
        for s in range(6)
    ]
    sd_con = float(np.std(con, ddof=1))
    sd_sin = float(np.std(sin, ddof=1))
    assert sd_con < sd_sin / 2.0, f"con puente {sd_con:.6f}, sin puente {sd_sin:.6f}"


def test_el_qmc_replicado_da_un_intervalo_de_confianza_valido():
    """La forma correcta de poner barras de error a un QMC.

    El precio verdadero debe caer dentro de unos pocos errores estandar del
    estimador replicado. Si el error estandar entre replicas estuviera mal
    calculado, esta prueba fallaria de forma sistematica.
    """
    res = price_qmc_replicated(S, K, R, Q, V, T, Payoff.CALL, 4096, 16, 16, seed=5)
    assert res.standard_error > 0.0
    assert abs(res.price - REF) < 4.0 * res.standard_error
    bajo, alto = res.confidence_interval_95
    assert bajo < res.price < alto


def test_el_mc_cubre_el_precio_verdadero_en_su_intervalo():
    res = price_mc(S, K, R, Q, V, T, Payoff.CALL, 200_000, 8, seed=3)
    assert abs(res.price - REF) < 4.0 * res.standard_error


def test_el_mc_es_reproducible_con_la_misma_semilla():
    a = price_mc(S, K, R, Q, V, T, Payoff.CALL, 8192, 8, seed=42)
    b = price_mc(S, K, R, Q, V, T, Payoff.CALL, 8192, 8, seed=42)
    assert a.price == b.price
    assert a.standard_error == b.standard_error


def test_el_puente_no_altera_el_precio_del_payoff_terminal():
    """Para un call europeo el puente cambia el orden, no la distribucion.

    Ambas construcciones producen el mismo browniano terminal, asi que el precio
    debe coincidir dentro del error de simulacion.
    """
    con = price_mc(S, K, R, Q, V, T, Payoff.CALL, 100_000, 16, seed=11, use_bridge=True)
    sin = price_mc(S, K, R, Q, V, T, Payoff.CALL, 100_000, 16, seed=11, use_bridge=False)
    assert abs(con.price - sin.price) < 4.0 * (con.standard_error + sin.standard_error)


@pytest.mark.parametrize(
    ("payoff", "comparacion"),
    [
        (Payoff.ASIAN_CALL, "menor"),
        (Payoff.LOOKBACK_CALL, "mayor"),
    ],
)
def test_ordenamiento_economico_de_los_payoffs(payoff, comparacion):
    """La asiatica vale menos que la europea; la lookback, mas.

    El promedio tiene menos varianza que el valor terminal, asi que la asiatica
    es mas barata. El maximo domina al valor terminal ruta por ruta, asi que la
    lookback es mas cara. Son desigualdades que no dependen del modelo.
    """
    exotica = price_qmc(S, K, R, Q, V, T, payoff, 16_384, 32, seed=9).price
    if comparacion == "menor":
        assert exotica < REF
    else:
        assert exotica > REF


def test_el_nucleo_no_modifica_el_arreglo_de_entrada():
    """El nucleo recibe una vista constante: no debe escribir en el buffer."""
    from pricing import _kernel

    z = sobol_normals(1024, 8, seed=1)
    copia = z.copy()
    _kernel.price_gbm(z, S, K, R, Q, V, T, int(Payoff.CALL), True, 0)
    np.testing.assert_array_equal(z, copia)


def test_el_nucleo_rechaza_arreglos_de_dimension_incorrecta():
    from pricing import _kernel

    with pytest.raises(ValueError, match="matriz"):
        _kernel.price_gbm(np.zeros(100), S, K, R, Q, V, T, 0, True, 0)


def test_el_nucleo_rechaza_parametros_fuera_de_dominio():
    from pricing import _kernel

    z = pseudorandom_normals(256, 4)
    with pytest.raises(ValueError, match="fuera de dominio"):
        _kernel.price_gbm(z, -1.0, K, R, Q, V, T, 0, True, 0)


# --------------------------------------------------------- griegas simuladas


def test_la_delta_pathwise_coincide_con_la_analitica():
    exacto = bs.greeks(S, K, R, Q, V, T)
    est = pathwise_greeks(S, K, R, Q, V, T, n_paths=400_000, seed=1)
    assert abs(est.delta - exacto.delta) < 4.0 * est.delta_stderr


def test_la_vega_pathwise_coincide_con_la_analitica():
    exacto = bs.greeks(S, K, R, Q, V, T)
    est = pathwise_greeks(S, K, R, Q, V, T, n_paths=400_000, seed=1)
    assert est.vega is not None and est.vega_stderr is not None
    assert abs(est.vega - exacto.vega) < 4.0 * est.vega_stderr


def test_el_estimador_pathwise_no_entrega_gamma():
    """La derivada del payoff de un call es una indicadora: su derivada no existe
    como funcion, asi que el estimador pathwise de gamma no esta definido."""
    est = pathwise_greeks(S, K, R, Q, V, T, n_paths=10_000, seed=1)
    assert est.gamma is None


def test_la_delta_y_la_gamma_por_razon_de_verosimilitud_coinciden_con_las_analiticas():
    exacto = bs.greeks(S, K, R, Q, V, T)
    est = likelihood_ratio_greeks(S, K, R, Q, V, T, n_paths=2_000_000, seed=2)
    assert est.gamma is not None and est.gamma_stderr is not None
    assert abs(est.delta - exacto.delta) < 4.0 * est.delta_stderr
    assert abs(est.gamma - exacto.gamma) < 4.0 * est.gamma_stderr


def test_el_estimador_pathwise_tiene_menos_varianza_que_el_de_verosimilitud():
    """El trade-off que decide cual usar.

    Pathwise gana en varianza cuando el payoff es diferenciable; razon de
    verosimilitud es la unica via cuando no lo es. Para la delta de un call en el
    dinero la ventaja medida es de un factor cercano a 2.4 en error estandar,
    equivalente a unas 6 veces menos rutas para la misma precision.
    """
    pw = pathwise_greeks(S, K, R, Q, V, T, n_paths=200_000, seed=3)
    lr = likelihood_ratio_greeks(S, K, R, Q, V, T, n_paths=200_000, seed=3)
    assert pw.delta_stderr < lr.delta_stderr / 2.0
