"""Benchmarks reproducibles del motor.

Produce las cifras que publica el README. Todas las semillas estan fijadas, asi
que la salida es reproducible.

Ejecutar con:

    python examples/benchmarks.py
"""

from __future__ import annotations

import time

import numpy as np

from pricing import (
    Payoff,
    aad_greeks,
    black_scholes as bs,
    gamma_curve,
    has_openmp,
    likelihood_ratio_greeks,
    pathwise_greeks,
    price_mc,
    price_qmc,
    price_qmc_replicated,
    solve_american,
    solve_european,
)

S, K, R, Q, V, T = 100.0, 100.0, 0.05, 0.02, 0.20, 1.0
SEMILLAS = range(8)


def seccion(titulo: str) -> None:
    print(f"\n{'=' * 72}\n{titulo}\n{'=' * 72}")


def bench_qmc_vs_mc() -> None:
    seccion("QMC con Sobol frente a Monte Carlo pseudoaleatorio")
    ref = bs.price(S, K, R, Q, V, T)
    print(f"referencia Black-Scholes: {ref:.8f}\n")
    print(f"{'rutas':>8} {'error MC':>12} {'error QMC':>12} {'ventaja':>10}")
    for n in (1024, 4096, 16_384, 65_536):
        err_mc = np.mean(
            [abs(price_mc(S, K, R, Q, V, T, Payoff.CALL, n, 16, seed=1000 + s).price - ref) for s in SEMILLAS]
        )
        err_qmc = np.mean(
            [abs(price_qmc(S, K, R, Q, V, T, Payoff.CALL, n, 16, seed=1000 + s).price - ref) for s in SEMILLAS]
        )
        print(f"{n:>8} {err_mc:>12.6f} {err_qmc:>12.6f} {err_mc / err_qmc:>9.0f}x")


def bench_puente_browniano() -> None:
    seccion("Puente browniano: ordenamiento de dimensiones por varianza")
    print("Opcion asiatica, 32 pasos. Dispersion entre 8 mezclados independientes.\n")
    print(f"{'rutas':>8} {'con puente':>13} {'sin puente':>13} {'ventaja':>10}")
    for n in (4096, 16_384, 65_536):
        con = [
            price_qmc(S, K, R, Q, V, T, Payoff.ASIAN_CALL, n, 32, seed=200 + s, use_bridge=True).price
            for s in SEMILLAS
        ]
        sin = [
            price_qmc(S, K, R, Q, V, T, Payoff.ASIAN_CALL, n, 32, seed=200 + s, use_bridge=False).price
            for s in SEMILLAS
        ]
        sd_con = float(np.std(con, ddof=1))
        sd_sin = float(np.std(sin, ddof=1))
        print(f"{n:>8} {sd_con:>13.6f} {sd_sin:>13.6f} {sd_sin / sd_con:>9.1f}x")
    print("\nSin el reordenamiento, Sobol gasta sus mejores dimensiones en los")
    print("incrementos de menor varianza y la ventaja sobre el MC casi desaparece.")


def bench_intervalo_valido() -> None:
    seccion("Intervalo de confianza valido para QMC")
    ref = bs.price(S, K, R, Q, V, T)
    crudo = price_qmc(S, K, R, Q, V, T, Payoff.CALL, 65_536, 16, seed=5)
    repl = price_qmc_replicated(S, K, R, Q, V, T, Payoff.CALL, 4096, 16, 16, seed=5)
    print(f"QMC crudo     : {crudo.price:.6f}  error real {abs(crudo.price - ref):.2e}")
    print(f"               error estandar reportado {crudo.standard_error:.6f} <- NO es valido")
    print("               (los puntos de Sobol no son independientes)\n")
    print(f"QMC replicado : {repl.price:.6f}  error real {abs(repl.price - ref):.2e}")
    print(f"               error estandar entre replicas {repl.standard_error:.6f} <- si es valido")
    bajo, alto = repl.confidence_interval_95
    print(f"               IC 95 % [{bajo:.6f}, {alto:.6f}]  contiene el valor real: "
          f"{bajo <= ref <= alto}")


def bench_rannacher() -> None:
    seccion("Arranque de Rannacher: la gamma de Crank-Nicolson puro es inservible")
    gamma_real = bs.greeks(S, K, R, 0.0, V, T).gamma
    print(f"gamma analitica en el dinero: {gamma_real:.8f}")
    print("Malla espacial fina (1200) y temporal burda (8 pasos), que es el regimen")
    print("en el que la inestabilidad de Crank-Nicolson se manifiesta.\n")
    print(f"{'pasos Rannacher':>16} {'err max gamma':>15} {'variacion total':>17}")
    for rs in (0, 1, 2, 4):
        res = solve_european(
            S, K, R, 0.0, V, T, n_space=1200, n_time=8, rannacher_steps=rs, return_grid=True
        )
        spots, gammas = gamma_curve(res, 80.0, 125.0)
        exactas = np.array([bs.greeks(float(x), K, R, 0.0, V, T).gamma for x in spots])
        err = gammas - exactas
        print(f"{rs:>16} {np.max(np.abs(err)):>15.6f} {np.sum(np.abs(np.diff(err))):>17.6f}")
    print(f"\nCon 0 pasos el error maximo supera a gamma en un factor de "
          f"{4.364081 / gamma_real:.0f}: la cifra no es utilizable.")


def bench_aad() -> None:
    seccion("AD en modo reverso frente a las griegas analiticas")
    exacto = bs.greeks(S, K, R, Q, V, T)
    ad = aad_greeks(S, K, R, Q, V, T)
    print(f"{'griega':>8} {'AAD':>16} {'analitica':>16} {'error relativo':>16}")
    for nombre in ("price", "delta", "vega", "rho", "theta"):
        a = ad[nombre]
        e = getattr(exacto, nombre)
        rel = abs(a - e) / max(abs(e), 1e-30)
        print(f"{nombre:>8} {a:>16.10f} {e:>16.10f} {rel:>16.2e}")
    print(f"\nnodos de la cinta: {int(ad['tape_size'])}")
    print("Seis derivadas en un solo recorrido hacia atras. Con diferencias finitas")
    print("harian falta seis revaluaciones completas del precio.")


def bench_griegas_simuladas() -> None:
    seccion("Estimadores de griegas por simulacion")
    exacto = bs.greeks(S, K, R, Q, V, T)
    pw = pathwise_greeks(S, K, R, Q, V, T, n_paths=500_000, seed=3)
    lr = likelihood_ratio_greeks(S, K, R, Q, V, T, n_paths=500_000, seed=3)
    print(f"delta analitica: {exacto.delta:.8f}")
    print(f"  pathwise      : {pw.delta:.6f} +/- {pw.delta_stderr:.6f}")
    print(f"  verosimilitud : {lr.delta:.6f} +/- {lr.delta_stderr:.6f}")
    print(f"  ventaja de pathwise en error estandar: {lr.delta_stderr / pw.delta_stderr:.1f}x")
    print(f"\ngamma analitica: {exacto.gamma:.8f}")
    print(f"  pathwise      : no existe (requeriria derivar una indicadora)")
    print(f"  verosimilitud : {lr.gamma:.6f} +/- {lr.gamma_stderr:.6f}")


def bench_americana() -> None:
    seccion("Opciones americanas")
    put_eu = bs.price(S, K, R, 0.0, V, T, False)
    put_am = solve_american(S, K, R, 0.0, V, T, 600, 600, is_call=False)
    call_eu = bs.price(S, K, R, 0.0, V, T, True)
    call_am = solve_american(S, K, R, 0.0, V, T, 600, 600, is_call=True)
    print(f"put  europeo {put_eu:.6f}   americano {put_am.price:.6f}   "
          f"prima de ejercicio {put_am.price - put_eu:+.6f}")
    print(f"call europeo {call_eu:.6f}   americano {call_am.price:.6f}   "
          f"diferencia         {call_am.price - call_eu:+.6f}")
    print("\nSin dividendos nunca conviene ejercer un call antes del vencimiento, asi")
    print("que su valor coincide con el europeo. El put si gana valor por ejercicio")
    print("anticipado: cobrar el strike antes permite invertirlo a la tasa libre.")


def bench_rendimiento() -> None:
    seccion("Rendimiento del nucleo nativo")
    print(f"OpenMP activo: {has_openmp()}\n")
    for n, pasos in ((262_144, 32), (262_144, 128)):
        t0 = time.perf_counter()
        res = price_qmc(S, K, R, Q, V, T, Payoff.ASIAN_CALL, n, pasos, seed=1)
        dt = time.perf_counter() - t0
        total = n * pasos
        print(f"{n:>8} rutas x {pasos:>4} pasos = {total / 1e6:>5.1f}M pasos "
              f"en {dt:>6.3f} s  ({total / dt / 1e6:>6.1f}M pasos/s)  precio {res.price:.6f}")


def main() -> None:
    bench_qmc_vs_mc()
    bench_puente_browniano()
    bench_intervalo_valido()
    bench_rannacher()
    bench_aad()
    bench_griegas_simuladas()
    bench_americana()
    bench_rendimiento()


if __name__ == "__main__":
    main()
