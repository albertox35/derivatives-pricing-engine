"""Validacion de Black-Scholes, la cinta de AD y el solver PDE.

Todo se contrasta contra la forma cerrada. La prueba mas importante del archivo
es la de Rannacher: demuestra con numeros que Crank-Nicolson puro produce una
gamma inservible, en lugar de afirmarlo.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricing import aad_greeks, black_scholes as bs, gamma_curve, has_openmp
from pricing import solve_american, solve_european

S, K, R, Q, V, T = 100.0, 100.0, 0.05, 0.02, 0.20, 1.0


# ------------------------------------------------------------- Black-Scholes


def test_precio_de_referencia():
    assert bs.price(S, K, R, Q, V, T) == pytest.approx(9.22700551, abs=1e-8)


def test_paridad_put_call():
    call = bs.price(S, K, R, Q, V, T, True)
    put = bs.price(S, K, R, Q, V, T, False)
    izquierda = call - put
    derecha = S * np.exp(-Q * T) - K * np.exp(-R * T)
    assert izquierda == pytest.approx(derecha, rel=1e-12)


def test_cotas_de_no_arbitraje():
    call = bs.price(S, K, R, Q, V, T, True)
    assert max(S * np.exp(-Q * T) - K * np.exp(-R * T), 0.0) <= call <= S * np.exp(-Q * T)


def test_el_precio_crece_con_la_volatilidad():
    precios = [bs.price(S, K, R, Q, vol, T) for vol in (0.1, 0.2, 0.3, 0.4)]
    assert all(b > a for a, b in zip(precios, precios[1:]))


@pytest.mark.parametrize(
    ("strike", "is_call", "vol"),
    [
        # Strike bajo: se cotiza el PUT, que es el lado fuera del dinero.
        (70.0, False, 0.15),
        (70.0, False, 0.35),
        (70.0, False, 0.80),
        # En el dinero: identificable en todo el rango de volatilidad.
        (100.0, True, 0.05),
        (100.0, True, 0.15),
        (100.0, True, 0.35),
        (100.0, True, 0.80),
        # Strike alto: se cotiza el CALL.
        (140.0, True, 0.15),
        (140.0, True, 0.35),
        (140.0, True, 0.80),
    ],
)
def test_volatilidad_implicita_recupera_la_de_entrada(strike, is_call, vol):
    """Se cotiza siempre el lado fuera del dinero, que es donde hay vega.

    Para el strike de 70 se usa el **put**, no el call: un call con strike 70 y
    spot 100 esta tan dentro del dinero que su vega es despreciable. Es la
    practica de mesa, no un atajo de la prueba.
    """
    objetivo = bs.price(S, strike, R, Q, vol, T, is_call)
    recuperada = bs.implied_volatility(objetivo, S, strike, R, Q, T, is_call)
    assert recuperada == pytest.approx(vol, rel=1e-8)


def test_volatilidad_implicita_rechaza_precios_sin_arbitraje():
    with pytest.raises(ValueError, match="fuera de las cotas de no-arbitraje"):
        bs.implied_volatility(500.0, S, K, R, Q, T)


def test_vega_es_identica_para_el_call_y_el_put_del_mismo_strike():
    """Consecuencia de la paridad put-call, y la razon de la prueba siguiente.

    Como la paridad relaciona call y put por terminos que no dependen de la
    volatilidad, al derivar respecto a sigma esos terminos desaparecen y las dos
    vegas coinciden exactamente.
    """
    for strike in (60.0, 100.0, 150.0):
        for vol in (0.05, 0.3):
            v_call = bs.greeks(S, strike, R, Q, vol, T, True).vega
            v_put = bs.greeks(S, strike, R, Q, vol, T, False).vega
            assert v_call == pytest.approx(v_put, rel=1e-12)


@pytest.mark.parametrize("is_call", [True, False])
def test_detecta_el_caso_no_identificable_por_ambos_lados(is_call):
    """Si vega es despreciable, ningun instrumento del strike es identificable.

    Es la consecuencia directa de que vega sea la misma para el call y el put.
    El mensaje de error tiene que decir que cambiar de lado no ayuda, en lugar de
    dar un consejo que no funciona.
    """
    objetivo = bs.price(S, 140.0, R, Q, 0.05, T, is_call)
    with pytest.raises(ValueError, match="no es identificable.*NO ayuda"):
        bs.implied_volatility(objetivo, S, 140.0, R, Q, T, is_call)


def test_un_strike_mas_cercano_al_dinero_si_es_identificable():
    """La salida que propone el mensaje de error efectivamente funciona."""
    objetivo = bs.price(S, 102.0, R, Q, 0.05, T, True)
    recuperada = bs.implied_volatility(objetivo, S, 102.0, R, Q, T, True)
    assert recuperada == pytest.approx(0.05, rel=1e-6)


def test_rechaza_parametros_fuera_de_dominio():
    with pytest.raises(ValueError, match="volatilidad"):
        bs.price(S, K, R, Q, 0.0, T)
    with pytest.raises(ValueError, match="vencimiento"):
        bs.price(S, K, R, Q, V, 0.0)
    with pytest.raises(ValueError, match="positivos"):
        bs.price(-1.0, K, R, Q, V, T)


# ------------------------------------------------ AD en modo reverso (AAD)


@pytest.mark.parametrize("is_call", [True, False])
def test_la_cinta_de_ad_reproduce_todas_las_griegas_analiticas(is_call):
    """La prueba central del nucleo nativo.

    Si la cinta estuviera mal construida --una parcial equivocada, un padre mal
    enlazado-- alguna de estas cinco derivadas no cuadraria. Que las cinco
    coincidan a 1e-9 es evidencia fuerte de que el recorrido reverso es correcto.
    """
    exacto = bs.greeks(S, K, R, Q, V, T, is_call)
    ad = aad_greeks(S, K, R, Q, V, T, is_call)

    assert ad["price"] == pytest.approx(exacto.price, rel=1e-12)
    assert ad["delta"] == pytest.approx(exacto.delta, rel=1e-9)
    assert ad["vega"] == pytest.approx(exacto.vega, rel=1e-9)
    assert ad["rho"] == pytest.approx(exacto.rho, rel=1e-9)
    assert ad["theta"] == pytest.approx(exacto.theta, rel=1e-9)


@pytest.mark.parametrize(
    ("spot", "strike", "vol", "maturity"),
    [
        (100.0, 60.0, 0.15, 0.25),  # muy dentro del dinero, corto plazo
        (100.0, 160.0, 0.45, 2.0),  # muy fuera del dinero, largo plazo
        (100.0, 100.0, 0.05, 5.0),  # volatilidad baja, muy largo plazo
        (3.5, 4.0, 0.9, 0.1),  # subyacente de nivel bajo, vol alta
    ],
)
def test_la_cinta_de_ad_es_exacta_en_regimenes_extremos(spot, strike, vol, maturity):
    exacto = bs.greeks(spot, strike, R, Q, vol, maturity)
    ad = aad_greeks(spot, strike, R, Q, vol, maturity)
    assert ad["delta"] == pytest.approx(exacto.delta, rel=1e-8, abs=1e-12)
    assert ad["vega"] == pytest.approx(exacto.vega, rel=1e-8, abs=1e-12)
    assert ad["theta"] == pytest.approx(exacto.theta, rel=1e-8, abs=1e-12)


def test_la_dual_delta_cumple_su_identidad():
    """La derivada respecto al strike de un call es ``-e^{-rT} N(d2)``."""
    from scipy.stats import norm

    ad = aad_greeks(S, K, R, Q, V, T, True)
    d1 = (np.log(S / K) + (R - Q + 0.5 * V**2) * T) / (V * np.sqrt(T))
    d2 = d1 - V * np.sqrt(T)
    esperado = -np.exp(-R * T) * norm.cdf(d2)
    assert ad["dual_delta"] == pytest.approx(float(esperado), rel=1e-9)


def test_el_costo_de_la_cinta_es_independiente_del_numero_de_parametros():
    """Seis derivadas en un solo recorrido, con una cinta de tamanio fijo.

    Es la propiedad que hace viable el modo reverso: con diferencias finitas
    harian falta seis revaluaciones completas.
    """
    ad = aad_greeks(S, K, R, Q, V, T)
    derivadas = ["delta", "dual_delta", "rho", "dividend_sensitivity", "vega", "theta"]
    assert all(k in ad for k in derivadas)
    assert 0 < ad["tape_size"] < 100


# ------------------------------------------------------------------ solver PDE


def test_el_solver_pde_converge_a_black_scholes():
    exacto = bs.price(S, K, R, Q, V, T)
    res = solve_european(S, K, R, Q, V, T, n_space=800, n_time=400)
    assert res.price == pytest.approx(exacto, abs=5e-4)


def test_el_error_del_pde_decrece_al_refinar_la_malla():
    exacto = bs.price(S, K, R, Q, V, T)
    errores = [
        abs(solve_european(S, K, R, Q, V, T, n_space=n, n_time=n // 2).price - exacto)
        for n in (100, 200, 400, 800)
    ]
    assert all(b < a for a, b in zip(errores, errores[1:])), errores


def test_las_griegas_del_pde_coinciden_con_las_analiticas():
    exacto = bs.greeks(S, K, R, Q, V, T)
    res = solve_european(S, K, R, Q, V, T, n_space=800, n_time=400)
    assert res.delta == pytest.approx(exacto.delta, abs=1e-4)
    assert res.gamma == pytest.approx(exacto.gamma, abs=1e-5)


def test_rannacher_elimina_la_oscilacion_de_crank_nicolson():
    """Prueba de la decision de diseno mas importante del solver.

    Con malla espacial fina y pocos pasos temporales, Crank-Nicolson puro excita
    los modos de alta frecuencia del kink del payoff y los propaga sin
    amortiguar. El precio apenas se nota; **gamma queda inservible**.

    Se mide la variacion total del error de gamma, que detecta oscilacion y no
    sesgo: un error grande pero suave da variacion total pequenia.
    """
    def variacion_total(rannacher_steps: int) -> tuple[float, float]:
        res = solve_european(
            S,
            K,
            R,
            0.0,
            V,
            T,
            n_space=1200,
            n_time=8,
            rannacher_steps=rannacher_steps,
            return_grid=True,
        )
        spots, gammas = gamma_curve(res, 80.0, 125.0)
        exactas = np.array([bs.greeks(float(x), K, R, 0.0, V, T).gamma for x in spots])
        error = gammas - exactas
        return float(np.sum(np.abs(np.diff(error)))), float(np.max(np.abs(error)))

    tv_sin, max_sin = variacion_total(0)
    tv_con, max_con = variacion_total(2)

    gamma_real = bs.greeks(S, K, R, 0.0, V, T).gamma

    # Sin Rannacher el error maximo supera el propio valor de gamma en dos
    # ordenes de magnitud: la cifra no es utilizable.
    assert max_sin > 100.0 * gamma_real
    # Con Rannacher el error cae por debajo del 1 % de gamma.
    assert max_con < 0.01 * gamma_real
    # Y la oscilacion se reduce en mas de tres ordenes de magnitud.
    assert tv_sin / tv_con > 1000.0


def test_el_put_americano_vale_mas_que_el_europeo():
    """Con tasa positiva el ejercicio anticipado tiene valor."""
    europeo = bs.price(S, K, R, 0.0, V, T, False)
    americano = solve_american(S, K, R, 0.0, V, T, n_space=600, n_time=600, is_call=False)
    assert americano.price > europeo
    assert americano.price - europeo > 0.1


def test_el_call_americano_sin_dividendos_coincide_con_el_europeo():
    """Nunca conviene ejercer un call anticipadamente si no hay dividendos."""
    europeo = bs.price(S, K, R, 0.0, V, T, True)
    americano = solve_american(S, K, R, 0.0, V, T, n_space=600, n_time=600, is_call=True)
    assert americano.price == pytest.approx(europeo, abs=5e-3)


def test_el_americano_nunca_vale_menos_que_su_payoff():
    for spot in (60.0, 80.0, 100.0, 120.0):
        res = solve_american(spot, K, R, 0.0, V, T, n_space=400, n_time=400, is_call=False)
        assert res.price >= max(K - spot, 0.0) - 1e-9


def test_el_pde_exige_malla_espacial_par():
    with pytest.raises(ValueError, match="par"):
        solve_european(S, K, R, Q, V, T, n_space=401, n_time=200)


def test_el_pde_rechaza_mallas_degeneradas():
    with pytest.raises(ValueError, match="demasiado burda"):
        solve_european(S, K, R, Q, V, T, n_space=8, n_time=200)


def test_el_perfil_de_gamma_requiere_haber_pedido_la_malla():
    res = solve_european(S, K, R, Q, V, T, n_space=200, n_time=100)
    with pytest.raises(ValueError, match="return_grid=True"):
        res.gamma_profile()


# ----------------------------------------------------------------- compilacion


def test_el_nucleo_reporta_si_tiene_openmp():
    assert isinstance(has_openmp(), bool)
