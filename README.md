# derivatives-pricing-engine

Motor de valuación de derivados con núcleo nativo **C++20 + pybind11**. Tres rutas
de cálculo, todas contrastadas contra Black-Scholes en forma cerrada:

- **Monte Carlo cuasi-aleatorio** con red de Sobol mezclada y puente browniano con
  ordenamiento de dimensiones por varianza.
- **Diferencias finitas** por Crank-Nicolson con arranque de Rannacher, europeas y
  americanas.
- **Diferenciación automática en modo reverso**, que entrega todas las griegas en
  un solo recorrido.

```python
from pricing import aad_greeks, price_qmc_replicated, solve_american, Payoff

# Todas las griegas en un recorrido, exactas a precisión de máquina
g = aad_greeks(spot=100, strike=100, rate=0.05, dividend=0.02, vol=0.2, maturity=1.0)
print(g["delta"], g["vega"], g["theta"])     # error relativo ~1e-16

# QMC con intervalo de confianza válido
r = price_qmc_replicated(100, 100, 0.05, 0.02, 0.2, 1.0, Payoff.ASIAN_CALL,
                         n_paths=4096, n_steps=32, n_replications=16)
print(r)                                      # precio +/- error estándar real

# Put americano: la prima de ejercicio anticipado
print(solve_american(100, 100, 0.05, 0.0, 0.2, 1.0, is_call=False).price)
```

---

## Resultados medidos

Semillas fijas, reproducible con `python examples/benchmarks.py`.

### Sobol frente a Monte Carlo pseudoaleatorio

Error absoluto medio frente a Black-Scholes, promediado sobre 8 semillas:

| Rutas | Error MC | Error QMC | Ventaja |
|---|---|---|---|
| 1 024 | 0.511891 | 0.005676 | **90×** |
| 4 096 | 0.200941 | 0.001403 | **143×** |
| 16 384 | 0.099906 | 0.000371 | **269×** |
| 65 536 | 0.034749 | 0.000115 | **302×** |

La ventaja **crece** con el número de rutas, que es la firma de la convergencia
cuasi-aleatoria: el error del QMC decae casi como `1/N` frente al `1/√N` del MC.

### El puente browniano no es decorativo

Opción asiática con 32 pasos. Dispersión del precio entre 8 mezclados
independientes:

| Rutas | Con puente | Sin puente | Ventaja |
|---|---|---|---|
| 4 096 | 0.002823 | 0.016487 | 5.8× |
| 16 384 | 0.000774 | 0.008749 | 11.3× |
| 65 536 | 0.000391 | 0.005535 | **14.2×** |

Una red de Sobol está mejor equidistribuida en sus **primeras** dimensiones. El
puente browniano construye primero el punto terminal, luego el medio, luego los
cuartos: así las dimensiones de mejor calidad se gastan en los grados de libertad
de mayor varianza. Con la construcción incremental se gastan en incrementos de
varianza pequeña y la ventaja del QMC casi desaparece. **Es el error clásico al
implementar Sobol**, y por eso el ordenamiento se mide en lugar de suponerse.

### El arranque de Rannacher: gamma inservible contra gamma correcta

Malla espacial fina (1200 nodos) y temporal burda (8 pasos), que es el régimen
donde se manifiesta la inestabilidad. Gamma analítica en el dinero: **0.0188**.

| Pasos de Rannacher | Error máximo de gamma | Variación total del error |
|---|---|---|
| **0** (Crank-Nicolson puro) | **4.364081** | **10.861592** |
| 1 | 0.000619 | 0.002831 |
| 2 (por omisión) | **0.000081** | **0.000222** |
| 4 | 0.000207 | 0.000626 |

Sin Rannacher el error de gamma es **233 veces el valor de gamma**. El precio
apenas se nota; la segunda derivada queda destruida.

La causa es concreta: Crank-Nicolson es de segundo orden pero **no es L-estable**.
Su factor de amplificación tiende a `−1` para los modos de alta frecuencia, no a
cero, así que el kink del payoff en el strike excita modos que oscilan de paso en
paso en lugar de amortiguarse. Partir los primeros pasos en medios pasos
totalmente implícitos los amortigua sin degradar el orden global.

Se mide la **variación total** del error, no su magnitud, porque es lo que
distingue oscilación de sesgo: un error grande pero suave da variación total
pequeña.

### AD en modo reverso contra las griegas analíticas

| Griega | AAD | Analítica | Error relativo |
|---|---|---|---|
| precio | 9.2270055082 | 9.2270055082 | 0 |
| delta | 0.5868511461 | 0.5868511461 | 5.7e-16 |
| vega | 37.9011575100 | 37.9011575100 | 0 |
| rho | 49.4581091053 | 49.4581091053 | 8.6e-16 |
| theta | −5.0893189140 | −5.0893189140 | 1.8e-16 |

**Precisión de máquina en las cinco**, con una cinta de 31 nodos. Seis derivadas
(incluida la *dual delta*, la sensibilidad al strike) en un solo recorrido hacia
atrás. Con diferencias finitas harían falta seis revaluaciones completas, y con
sesgo.

Esa propiedad —costo independiente del número de parámetros— es lo que hace
viable calcular sensibilidades de un portafolio con cientos de factores de
riesgo, que es el caso de XVA.

### Estimadores de griegas por simulación

| Griega | Pathwise | Razón de verosimilitud | Analítica |
|---|---|---|---|
| delta | 0.586948 ± 0.000812 | 0.587629 ± 0.001958 | 0.5868511 |
| gamma | **no existe** | 0.018938 ± 0.000183 | 0.0189506 |

Pathwise tiene **2.4× menos error estándar** en delta, equivalente a ~6 veces
menos rutas para la misma precisión. Pero **no puede dar gamma**: la derivada del
payoff de un call es una función indicadora, cuya derivada es una delta de Dirac
que el estimador no representa. Ahí la razón de verosimilitud es la única vía.
Saber cuál falla y dónde es el punto del módulo.

### Americanas

| | Europeo | Americano | Diferencia |
|---|---|---|---|
| put (q = 0) | 5.573526 | 6.088938 | **+0.515412** |
| call (q = 0) | 10.450584 | 10.450221 | −0.000363 |

Las dos propiedades esperadas: el put gana valor por ejercicio anticipado —cobrar
el strike antes permite invertirlo—, y el call sin dividendos **no**, así que su
valor coincide con el europeo salvo error de malla.

### Rendimiento

OpenMP activo. Opción asiática, un solo proceso:

| Rutas × pasos | Tiempo | Rendimiento |
|---|---|---|
| 262 144 × 32 = 8.4M pasos | 0.617 s | 13.6M pasos/s |
| 262 144 × 128 = 33.6M pasos | 2.735 s | 12.3M pasos/s |

El costo está dominado por la exponencial por paso, que es irreducible sin
aproximarla. No se reporta un promedio de latencia porque aquí la métrica
relevante es el *throughput*.

---

## Qué no hace

- **No hay kernel CUDA, y eso es deliberado.** No se puede escribir y *verificar*
  código CUDA sin una GPU, y un repositorio que anuncia un kernel no probado vale
  menos que uno que declara su alcance. El núcleo CPU se paraleliza con OpenMP. La
  ruta a GPU existe conceptualmente —el Monte Carlo es vergonzosamente paralelo—
  pero no está implementada ni se finge que lo esté.
- **No hay nada de baja latencia.** Este repositorio resuelve un problema de
  *throughput*, no de latencia. Son objetivos de optimización distintos: en
  latencia de microsegundos no hay GPU en el camino crítico, porque el viaje de
  ida y vuelta por PCIe (~10 µs) excede el presupuesto completo. El componente de
  baja latencia vive en `market-data-pipeline`, que es su lugar.
- **La secuencia de Sobol se toma de SciPy.** `scipy.stats.qmc.Sobol` implementa
  los números de dirección de Joe-Kuo, validados hasta 21 201 dimensiones.
  Reimplementarla sería introducir riesgo sin ganancia: el costo del Monte Carlo
  está en construir trayectorias y evaluar payoffs, y ese bucle sí está en C++.
- **Un solo modelo de dinámica: movimiento browniano geométrico.** Heston, modelos
  de volatilidad local y SABR están en `volatility-surface-calibration`.
- **Payoffs limitados**: europeo, asiático aritmético y lookback. No hay barreras,
  ni cestas, ni Bermudas por Longstaff-Schwartz (esa técnica vive en
  `xva-engine`, que es donde hace falta).
- **El solver PDE es unidimensional.** Para multiactivo hacen falta esquemas ADI
  (Craig-Sneyd, Hundsdorfer-Verwer), no implementados.
- **La americana se resuelve por proyección explícita**, `v = max(v, payoff)` tras
  cada paso, no por PSOR sobre el problema de complementariedad lineal. Es
  suficientemente preciso con malla fina y mucho más simple.

---

## Alcance y decisiones de diseño

**«Latencia cero» no existe; zero-copy sí.** El núcleo recibe los arreglos de
NumPy por el protocolo buffer, sin copiarlos, y libera el GIL durante todo el
cómputo. Eso es medible y verificable; una «latencia cero» sería físicamente
imposible. Las funciones públicas de `qmc` devuelven arreglos `float64`
contiguos en C precisamente para que la frontera no introduzca una copia, y hay
una prueba que lo exige.

**C++20, no C++ y Rust.** Escribir el mismo núcleo en dos lenguajes de sistemas no
es una característica, es deuda técnica duplicada. Se eligió C++ porque el
ecosistema quant existente lo es, y pybind11 porque está más maduro que PyO3 para
numérico con buffers de NumPy.

**La reducción paralela es determinista, y la varianza se acumula con Welford.**
Esto empezó como un fallo de CI: la prueba de reproducibilidad pasaba en Windows
y fallaba en Linux, con el **precio idéntico bit a bit** y el error estándar
distinto en 1 ULP. Dos defectos encadenados:

1. Una reducción `reduction(+:...)` de OpenMP **no es reproducible**: el árbol de
   combinación depende del número de hilos que el runtime decida usar, que puede
   variar entre ejecuciones de la misma máquina.
2. La fórmula de varianza de libro de texto, `(Σx² − n·media²)/(n−1)`, resta dos
   cantidades grandes y casi iguales. Con payoffs del orden de 10 y desviación
   del orden de 0.15, `Σx²` es unos cuatro órdenes de magnitud mayor que el
   resultado buscado, así que **amplifica** esa diferencia de 1 ULP por
   cancelación catastrófica hasta hacerla visible.

El segundo es el defecto real; el primero solo lo destapó. Ahora el núcleo
acumula en un **número fijo de bloques** —independiente del número de hilos— con
el algoritmo de Welford, y los combina en orden de índice con la fórmula de Chan,
Golub y LeVeque. El resultado es **idéntico bit a bit con 1, 2, 4, 8 y 16
hilos**, y hay una prueba que lo exige. El estimador de varianza se contrasta
además contra NumPy por una vía independiente, con acuerdo a 1e-12.

Es exactamente el determinismo que el README reclama como requisito de
auditoría, ahora verificado frente al paralelismo y no solo frente a repetir la
misma llamada.

**El `standard_error` del QMC crudo se devuelve, pero se declara inválido.** Los
puntos de una red de Sobol no son independientes, así que la varianza muestral
sobre las rutas no estima el error del estimador. Para un intervalo de confianza
legítimo está `price_qmc_replicated`, que promedia mezclados independientes y
calcula el error **entre** réplicas. El benchmark muestra la diferencia: 0.054
reportado por el QMC crudo frente a 0.0004 real.

**Sobol exige potencia de dos, y se hace cumplir.** Las propiedades de
equidistribución se dan en bloques de `2^m`; pedir 10 000 puntos rompe el balance
y degrada la convergencia en silencio. La función lanza una excepción con el
tamaño correcto sugerido, y permite salirse de forma explícita.

**La volatilidad implícita puede no existir, y se dice.** Cuando vega es
despreciable el problema no es identificable: muchas sigmas reproducen el mismo
precio. La función lo detecta y lanza una excepción en lugar de devolver el primer
valor que converja. Y el mensaje es correcto en un punto donde es fácil
equivocarse: por paridad put-call, **vega es idéntica para el call y el put del
mismo strike**, así que cambiar de instrumento no ayuda. Las mesas cotizan el lado
fuera del dinero por acondicionamiento del precio —el valor temporal es una
fracción mayor de lo cotizado—, no porque tenga más vega.

**El repositorio es autocontenido.** No depende de una librería núcleo compartida.
La contrapartida es duplicación numérica entre repositorios; la ventaja, que este
se clona, compila y evalúa por sí solo.

---

## Instalación

Requiere un compilador con C++20. El flag de OpenMP se añade de forma condicional:
si no está disponible, la extensión compila igual y corre en un solo hilo.

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

```bash
python examples/benchmarks.py
```

Probado con MSVC 14.51 (Visual Studio 2026 Build Tools) y GCC. Si setuptools no
detecta una instalación reciente de Visual Studio, preconfigurar el entorno:

```bash
call "...\VC\Auxiliary\Build\vcvars64.bat" && set DISTUTILS_USE_SDK=1 && pip install -e .
```

---

## Referencias

- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
  Capítulo 7 para griegas por simulación, capítulo 5 para QMC.
- Joe, S. & Kuo, F. (2008). *Constructing Sobol sequences with better
  two-dimensional projections*. SIAM Journal on Scientific Computing 30(5).
- Owen, A. (1997). *Scrambled net variance for integrals of smooth functions*.
  Annals of Statistics 25(4).
- Caflisch, R., Morokoff, W. & Owen, A. (1997). *Valuation of mortgage-backed
  securities using Brownian bridges to reduce effective dimension*. Journal of
  Computational Finance 1(1).
- Rannacher, R. (1984). *Finite element solution of diffusion problems with
  irregular data*. Numerische Mathematik 43(2).
- Giles, M. & Glasserman, P. (2006). *Smoking adjoints: fast Monte Carlo greeks*.
  Risk Magazine 19(1).
- Capriotti, L. (2011). *Fast Greeks by algorithmic differentiation*. Journal of
  Computational Finance 14(3).
- Broadie, M. & Glasserman, P. (1996). *Estimating security price derivatives
  using simulation*. Management Science 42(2).

## Licencia

MIT. Ver [LICENSE](LICENSE).
