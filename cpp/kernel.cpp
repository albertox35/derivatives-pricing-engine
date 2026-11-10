// Nucleo nativo del motor de valuacion.
//
// Tres piezas viven aqui, y solo tres, porque son las unicas donde el codigo
// nativo gana algo real:
//
//   1. Construccion de trayectorias por puente browniano y evaluacion del
//      payoff. Es el bucle caliente: O(rutas x pasos) operaciones con muy poca
//      aritmetica por elemento, donde el costo por iteracion del interprete
//      domina.
//   2. Solver PDE de Crank-Nicolson con arranque de Rannacher. El algoritmo de
//      Thomas es inherentemente secuencial en el espacio y se repite en cada
//      paso temporal.
//   3. Cinta de diferenciacion automatica en modo reverso, que requiere
//      registrar cada operacion elemental. En Python el sobrecosto por nodo
//      haria la tecnica inutil.
//
// Lo que NO esta aqui es deliberado: la generacion de la secuencia de Sobol usa
// scipy.stats.qmc, que implementa los numeros de direccion de Joe-Kuo probados
// hasta 21201 dimensiones. Reimplementarla seria introducir riesgo sin ganancia:
// el costo esta en las trayectorias, no en generar los puntos.
//
// Convenciones de la interfaz:
//   - Los arreglos se reciben por el protocolo buffer de NumPy sin copiar.
//   - El GIL se libera durante todo el computo.
//   - Ninguna funcion asigna memoria dentro del bucle sobre rutas.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------- utilidades

// Vista de solo lectura sobre un buffer contiguo de NumPy, sin copia.
struct ConstView {
    const double* data;
    std::size_t size;
};

ConstView as_view(const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
    auto buf = a.request();
    return ConstView{static_cast<const double*>(buf.ptr), static_cast<std::size_t>(buf.size)};
}

// ------------------------------------------------- puente browniano (bridge)

// Orden de construccion del puente browniano.
//
// El puente construye primero el punto final, luego el medio, luego los cuartos,
// y asi sucesivamente. El efecto es concentrar la varianza en las primeras
// dimensiones de la secuencia cuasi-aleatoria.
//
// Esto es lo que hace que Sobol supere al Monte Carlo pseudoaleatorio: una
// secuencia de baja discrepancia tiene mejor equidistribucion en sus
// dimensiones iniciales, asi que conviene gastarlas en los grados de libertad
// que mas aportan a la varianza. Sin este reordenamiento el QMC rinde
// practicamente lo mismo que el MC, que es el error clasico al implementar
// Sobol.
struct BridgeSchedule {
    std::vector<std::size_t> point;   // indice temporal que se construye
    std::vector<std::size_t> left;    // extremo izquierdo conocido (0 = t_0)
    std::vector<std::size_t> right;   // extremo derecho conocido
    std::vector<double> left_weight;
    std::vector<double> right_weight;
    std::vector<double> stddev;
};

BridgeSchedule build_bridge_schedule(std::size_t n_steps) {
    if (n_steps == 0) {
        throw std::invalid_argument("n_steps debe ser positivo");
    }
    BridgeSchedule s;
    s.point.reserve(n_steps);
    s.left.reserve(n_steps);
    s.right.reserve(n_steps);
    s.left_weight.reserve(n_steps);
    s.right_weight.reserve(n_steps);
    s.stddev.reserve(n_steps);

    // map[i] indica si el punto temporal i+1 ya fue construido.
    std::vector<char> done(n_steps + 1, 0);
    done[0] = 1;  // t_0 = 0 se conoce siempre

    // Primera dimension: el punto terminal.
    s.point.push_back(n_steps);
    s.left.push_back(0);
    s.right.push_back(0);  // sin extremo derecho: es un incremento puro
    s.left_weight.push_back(1.0);
    s.right_weight.push_back(0.0);
    s.stddev.push_back(std::sqrt(static_cast<double>(n_steps)));
    done[n_steps] = 1;

    // Resto: subdivision binaria, siempre el intervalo pendiente mas largo.
    for (std::size_t k = 1; k < n_steps; ++k) {
        // Localizar el hueco mas largo entre puntos ya construidos.
        std::size_t best_left = 0, best_right = 0, best_len = 0;
        std::size_t left = 0;
        for (std::size_t i = 1; i <= n_steps; ++i) {
            if (!done[i]) continue;
            const std::size_t len = i - left;
            if (len > best_len) {
                best_len = len;
                best_left = left;
                best_right = i;
            }
            left = i;
        }
        if (best_len < 2) {
            throw std::logic_error("el calendario del puente browniano quedo inconsistente");
        }
        const std::size_t mid = best_left + best_len / 2;
        const double dt_left = static_cast<double>(mid - best_left);
        const double dt_right = static_cast<double>(best_right - mid);
        const double total = dt_left + dt_right;

        s.point.push_back(mid);
        s.left.push_back(best_left);
        s.right.push_back(best_right);
        s.left_weight.push_back(dt_right / total);
        s.right_weight.push_back(dt_left / total);
        s.stddev.push_back(std::sqrt(dt_left * dt_right / total));
        done[mid] = 1;
    }
    return s;
}

// Construye una trayectoria del movimiento browniano estandar en el reticulado
// {0, dt, 2dt, ..., T} a partir de n_steps normales estandar, en orden de
// puente. Escribe en w[0..n_steps], con w[0] = 0.
inline void fill_brownian_path(const BridgeSchedule& s, const double* normals, double dt,
                               double* w, std::size_t n_steps) {
    const double sqrt_dt = std::sqrt(dt);
    w[0] = 0.0;
    // Dimension 0: punto terminal.
    w[s.point[0]] = sqrt_dt * s.stddev[0] * normals[0];
    for (std::size_t k = 1; k < n_steps; ++k) {
        const double wl = w[s.left[k]];
        const double wr = w[s.right[k]];
        w[s.point[k]] = s.left_weight[k] * wl + s.right_weight[k] * wr +
                        sqrt_dt * s.stddev[k] * normals[k];
    }
}

// --------------------------------------------------------------- payoffs

enum class PayoffKind { Call = 0, Put = 1, AsianCall = 2, AsianPut = 3, LookbackCall = 4 };

}  // namespace

// ------------------------------------------------- Monte Carlo sobre GBM

// Valua una opcion bajo movimiento browniano geometrico por Monte Carlo.
//
// `normals` es (n_paths, n_steps) y se consume sin copiar. Las normales las
// genera el lado Python, que es donde vive el control de la secuencia (Sobol
// mezclado o pseudoaleatoria): asi el nucleo no impone la fuente de aleatoriedad
// y la misma rutina sirve para MC y QMC.
py::dict price_gbm(py::array_t<double, py::array::c_style | py::array::forcecast> normals,
                   double spot, double strike, double rate, double dividend, double vol,
                   double maturity, int payoff_code, bool use_bridge, int n_threads) {
    auto buf = normals.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("normals debe ser una matriz (n_paths, n_steps)");
    }
    const std::size_t n_paths = static_cast<std::size_t>(buf.shape[0]);
    const std::size_t n_steps = static_cast<std::size_t>(buf.shape[1]);
    if (n_paths == 0 || n_steps == 0) {
        throw std::invalid_argument("se requieren rutas y pasos positivos");
    }
    if (spot <= 0.0 || strike < 0.0 || vol < 0.0 || maturity <= 0.0) {
        throw std::invalid_argument("parametros de mercado fuera de dominio");
    }
    const auto kind = static_cast<PayoffKind>(payoff_code);
    const double* z = static_cast<const double*>(buf.ptr);

    const double dt = maturity / static_cast<double>(n_steps);
    const double drift = (rate - dividend - 0.5 * vol * vol) * dt;
    const double discount = std::exp(-rate * maturity);

    double sum = 0.0;
    double sum_sq = 0.0;

    {
        py::gil_scoped_release release;  // el computo no toca objetos de Python
        const BridgeSchedule schedule = build_bridge_schedule(n_steps);

#ifdef _OPENMP
        if (n_threads > 0) omp_set_num_threads(n_threads);
#endif

#ifdef _OPENMP
#pragma omp parallel reduction(+ : sum, sum_sq)
#endif
        {
            // Buffers por hilo, asignados una sola vez fuera del bucle de rutas.
            std::vector<double> w(n_steps + 1, 0.0);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(n_paths); ++p) {
                const double* row = z + static_cast<std::size_t>(p) * n_steps;

                if (use_bridge) {
                    fill_brownian_path(schedule, row, dt, w.data(), n_steps);
                } else {
                    // Construccion incremental secuencial.
                    const double sqrt_dt = std::sqrt(dt);
                    w[0] = 0.0;
                    for (std::size_t i = 0; i < n_steps; ++i) {
                        w[i + 1] = w[i] + sqrt_dt * row[i];
                    }
                }

                // Trayectoria del subyacente y acumuladores del payoff.
                double running_sum = 0.0;
                double running_max = spot;
                double s_t = spot;
                for (std::size_t i = 1; i <= n_steps; ++i) {
                    s_t = spot * std::exp(drift * static_cast<double>(i) + vol * w[i]);
                    running_sum += s_t;
                    running_max = std::max(running_max, s_t);
                }
                const double average = running_sum / static_cast<double>(n_steps);

                double payoff = 0.0;
                switch (kind) {
                    case PayoffKind::Call:
                        payoff = std::max(s_t - strike, 0.0);
                        break;
                    case PayoffKind::Put:
                        payoff = std::max(strike - s_t, 0.0);
                        break;
                    case PayoffKind::AsianCall:
                        payoff = std::max(average - strike, 0.0);
                        break;
                    case PayoffKind::AsianPut:
                        payoff = std::max(strike - average, 0.0);
                        break;
                    case PayoffKind::LookbackCall:
                        payoff = std::max(running_max - strike, 0.0);
                        break;
                }
                const double value = discount * payoff;
                sum += value;
                sum_sq += value * value;
            }
        }
    }

    const double n = static_cast<double>(n_paths);
    const double mean = sum / n;
    // Varianza muestral; con n = 1 no esta definida.
    const double var = n > 1.0 ? std::max((sum_sq - n * mean * mean) / (n - 1.0), 0.0) : 0.0;

    py::dict out;
    out["price"] = mean;
    out["standard_error"] = std::sqrt(var / n);
    out["n_paths"] = n_paths;
    out["n_steps"] = n_steps;
    return out;
}

// ------------------------------------- PDE: Crank-Nicolson con Rannacher

// Resuelve la ecuacion de Black-Scholes por diferencias finitas en la variable
// log-espacial, con malla uniforme y condiciones de frontera de Dirichlet.
//
// El arranque de Rannacher es la parte que importa. Crank-Nicolson es de segundo
// orden en tiempo pero NO es L-estable: con una condicion inicial discontinua en
// su derivada --el kink del payoff en el strike-- produce oscilaciones espurias
// que contaminan gamma y theta cerca del dinero. Rannacher lo corrige
// sustituyendo los primeros pasos por medios pasos totalmente implicitos, que si
// amortiguan los modos de alta frecuencia, sin degradar el orden global.
//
// Omitir Rannacher es el error clasico de un solver de Crank-Nicolson, y se
// manifiesta justo donde mas duele: en las griegas de segundo orden.
py::dict solve_pde(double spot, double strike, double rate, double dividend, double vol,
                   double maturity, int n_space, int n_time, bool is_call, bool american,
                   int rannacher_steps, double n_stddev, bool return_grid) {
    if (n_space < 16 || n_time < 4) {
        throw std::invalid_argument("la malla es demasiado burda (n_space >= 16, n_time >= 4)");
    }
    if (spot <= 0.0 || strike <= 0.0 || vol <= 0.0 || maturity <= 0.0) {
        throw std::invalid_argument("parametros de mercado fuera de dominio");
    }
    if (rannacher_steps < 0 || rannacher_steps > n_time) {
        throw std::invalid_argument("rannacher_steps fuera de rango");
    }

    double price = 0.0, delta = 0.0, gamma = 0.0;
    std::vector<double> grid_s, grid_v;

    {
        py::gil_scoped_release release;

        // Malla en x = log(S), centrada en log(spot).
        const double x0 = std::log(spot);
        const double half_width = n_stddev * vol * std::sqrt(maturity);
        const double x_min = x0 - half_width;
        const double x_max = x0 + half_width;
        const std::size_t m = static_cast<std::size_t>(n_space);
        const double dx = (x_max - x_min) / static_cast<double>(m);
        const double dt = maturity / static_cast<double>(n_time);

        std::vector<double> x(m + 1), v(m + 1), payoff(m + 1);
        for (std::size_t i = 0; i <= m; ++i) {
            x[i] = x_min + static_cast<double>(i) * dx;
            const double s = std::exp(x[i]);
            payoff[i] = is_call ? std::max(s - strike, 0.0) : std::max(strike - s, 0.0);
            v[i] = payoff[i];
        }

        // Coeficientes del operador espacial en x (constantes: malla uniforme).
        const double a = 0.5 * vol * vol;
        const double b = rate - dividend - 0.5 * vol * vol;
        const double lower_op = a / (dx * dx) - b / (2.0 * dx);
        const double diag_op = -2.0 * a / (dx * dx) - rate;
        const double upper_op = a / (dx * dx) + b / (2.0 * dx);

        std::vector<double> lo(m + 1), di(m + 1), up(m + 1), rhs(m + 1), cprime(m + 1),
            dprime(m + 1), v_old(m + 1);

        const double s_lo = std::exp(x[0]);
        const double s_hi = std::exp(x[m]);

        // Valor de frontera de Dirichlet a tiempo restante `tau`.
        //
        // Usar el tiempo restante hasta el vencimiento, y no el tamanio del paso,
        // es esencial: en el borde superior un call vale asintoticamente
        // S*e^{-q*tau} - K*e^{-r*tau}, que depende de cuanto falta para expirar.
        auto boundary = [&](double tau) {
            std::pair<double, double> bc;
            if (is_call) {
                bc.first = 0.0;
                bc.second = s_hi * std::exp(-dividend * tau) - strike * std::exp(-rate * tau);
                if (american) bc.second = std::max(bc.second, s_hi - strike);
            } else {
                bc.first = strike * std::exp(-rate * tau) - s_lo * std::exp(-dividend * tau);
                if (american) bc.first = std::max(bc.first, strike - s_lo);
                bc.second = 0.0;
            }
            return bc;
        };

        // Un paso implicito con peso theta:
        //   (I - theta*dt*L) v^{n+1} = (I + (1-theta)*dt*L) v^n
        // theta = 1 -> totalmente implicito; theta = 1/2 -> Crank-Nicolson.
        //
        // `tau_old` y `tau_new` son el tiempo restante antes y despues del paso.
        auto step = [&](double step_dt, double theta, double tau_old, double tau_new) {
            // La parte explicita tiene que leer las fronteras ANTIGUAS, asi que
            // se congela el estado previo antes de tocar nada.
            std::copy(v.begin(), v.end(), v_old.begin());
            const auto bc_old = boundary(tau_old);
            const auto bc_new = boundary(tau_new);
            v_old[0] = bc_old.first;
            v_old[m] = bc_old.second;

            for (std::size_t i = 1; i < m; ++i) {
                rhs[i] = v_old[i] +
                         (1.0 - theta) * step_dt *
                             (lower_op * v_old[i - 1] + diag_op * v_old[i] + upper_op * v_old[i + 1]);
            }
            for (std::size_t i = 1; i < m; ++i) {
                lo[i] = -theta * step_dt * lower_op;
                di[i] = 1.0 - theta * step_dt * diag_op;
                up[i] = -theta * step_dt * upper_op;
            }
            // Las fronteras NUEVAS entran en el lado implicito.
            rhs[1] -= lo[1] * bc_new.first;
            rhs[m - 1] -= up[m - 1] * bc_new.second;

            // Algoritmo de Thomas.
            cprime[1] = up[1] / di[1];
            dprime[1] = rhs[1] / di[1];
            for (std::size_t i = 2; i < m; ++i) {
                const double denom = di[i] - lo[i] * cprime[i - 1];
                cprime[i] = up[i] / denom;
                dprime[i] = (rhs[i] - lo[i] * dprime[i - 1]) / denom;
            }
            v[m - 1] = dprime[m - 1];
            for (std::size_t i = m - 1; i-- > 1;) {
                v[i] = dprime[i] - cprime[i] * v[i + 1];
            }
            v[0] = bc_new.first;
            v[m] = bc_new.second;

            if (american) {
                // Proyeccion sobre la restriccion de ejercicio anticipado.
                for (std::size_t i = 0; i <= m; ++i) {
                    v[i] = std::max(v[i], payoff[i]);
                }
            }
        };

        // Arranque de Rannacher: los primeros pasos se parten en dos medios
        // pasos totalmente implicitos, que si amortiguan los modos de alta
        // frecuencia introducidos por el kink del payoff.
        double tau = 0.0;
        for (int n = 0; n < rannacher_steps; ++n) {
            step(0.5 * dt, 1.0, tau, tau + 0.5 * dt);
            tau += 0.5 * dt;
            step(0.5 * dt, 1.0, tau, tau + 0.5 * dt);
            tau += 0.5 * dt;
        }
        for (int n = rannacher_steps; n < n_time; ++n) {
            step(dt, 0.5, tau, tau + dt);
            tau += dt;
        }

        // Lectura en el spot: el nodo central es exactamente log(spot).
        const std::size_t ic = m / 2;
        // Interpolacion cuadratica y diferencias centradas para las griegas.
        const double s_c = std::exp(x[ic]);
        price = v[ic];
        const double dv_dx = (v[ic + 1] - v[ic - 1]) / (2.0 * dx);
        const double d2v_dx2 = (v[ic + 1] - 2.0 * v[ic] + v[ic - 1]) / (dx * dx);
        // Cambio de variable de x = log S a S.
        delta = dv_dx / s_c;
        gamma = (d2v_dx2 - dv_dx) / (s_c * s_c);

        if (return_grid) {
            grid_s.resize(m + 1);
            grid_v.resize(m + 1);
            for (std::size_t i = 0; i <= m; ++i) {
                grid_s[i] = std::exp(x[i]);
                grid_v[i] = v[i];
            }
        }
    }

    py::dict out;
    out["price"] = price;
    out["delta"] = delta;
    out["gamma"] = gamma;
    if (return_grid) {
        out["spot_grid"] = py::array_t<double>(grid_s.size(), grid_s.data());
        out["value_grid"] = py::array_t<double>(grid_v.size(), grid_v.data());
    }
    return out;
}

// --------------------------------- diferenciacion automatica en modo reverso

namespace {

// Cinta de AD en modo reverso.
//
// Cada operacion elemental deja un nodo con hasta dos padres y las derivadas
// parciales locales. Al propagar los adjuntos hacia atras se obtienen TODAS las
// derivadas en un solo recorrido, a un costo de 3 a 5 veces el de la evaluacion
// directa, con independencia del numero de parametros.
//
// Esa propiedad es la que importa: calcular griegas por diferencias finitas
// cuesta una revaluacion completa por parametro, y en un portafolio con cientos
// de factores de riesgo --el caso de XVA-- eso es inviable.
class Tape {
   public:
    struct Node {
        std::size_t parent0;
        std::size_t parent1;
        double partial0;
        double partial1;
    };

    std::size_t push_leaf() {
        nodes_.push_back(Node{kNoParent, kNoParent, 0.0, 0.0});
        return nodes_.size() - 1;
    }

    std::size_t push_unary(std::size_t parent, double partial) {
        nodes_.push_back(Node{parent, kNoParent, partial, 0.0});
        return nodes_.size() - 1;
    }

    std::size_t push_binary(std::size_t p0, double d0, std::size_t p1, double d1) {
        nodes_.push_back(Node{p0, p1, d0, d1});
        return nodes_.size() - 1;
    }

    // Propagacion hacia atras desde la raiz.
    std::vector<double> backward(std::size_t root) const {
        std::vector<double> adjoint(nodes_.size(), 0.0);
        adjoint[root] = 1.0;
        for (std::size_t i = nodes_.size(); i-- > 0;) {
            const double a = adjoint[i];
            if (a == 0.0) continue;
            const Node& nd = nodes_[i];
            if (nd.parent0 != kNoParent) adjoint[nd.parent0] += a * nd.partial0;
            if (nd.parent1 != kNoParent) adjoint[nd.parent1] += a * nd.partial1;
        }
        return adjoint;
    }

    std::size_t size() const { return nodes_.size(); }

    static constexpr std::size_t kNoParent = static_cast<std::size_t>(-1);

   private:
    std::vector<Node> nodes_;
};

// Numero dual con cinta: valor mas indice en la cinta.
struct ADouble {
    double value;
    std::size_t index;
};

Tape* g_tape = nullptr;

ADouble leaf(double v) { return ADouble{v, g_tape->push_leaf()}; }

ADouble operator+(const ADouble& a, const ADouble& b) {
    return ADouble{a.value + b.value, g_tape->push_binary(a.index, 1.0, b.index, 1.0)};
}
ADouble operator-(const ADouble& a, const ADouble& b) {
    return ADouble{a.value - b.value, g_tape->push_binary(a.index, 1.0, b.index, -1.0)};
}
ADouble operator*(const ADouble& a, const ADouble& b) {
    return ADouble{a.value * b.value,
                   g_tape->push_binary(a.index, b.value, b.index, a.value)};
}
ADouble operator/(const ADouble& a, const ADouble& b) {
    const double inv = 1.0 / b.value;
    return ADouble{a.value * inv,
                   g_tape->push_binary(a.index, inv, b.index, -a.value * inv * inv)};
}
ADouble operator*(double c, const ADouble& a) {
    return ADouble{c * a.value, g_tape->push_unary(a.index, c)};
}
ADouble operator-(double c, const ADouble& a) {
    return ADouble{c - a.value, g_tape->push_unary(a.index, -1.0)};
}
ADouble ad_log(const ADouble& a) {
    return ADouble{std::log(a.value), g_tape->push_unary(a.index, 1.0 / a.value)};
}
ADouble ad_exp(const ADouble& a) {
    const double e = std::exp(a.value);
    return ADouble{e, g_tape->push_unary(a.index, e)};
}
ADouble ad_sqrt(const ADouble& a) {
    const double r = std::sqrt(a.value);
    return ADouble{r, g_tape->push_unary(a.index, 0.5 / r)};
}
// Normal estandar acumulada mediante erfc, con derivada analitica.
ADouble ad_norm_cdf(const ADouble& a) {
    const double v = 0.5 * std::erfc(-a.value * 0.7071067811865476);
    const double pdf = 0.3989422804014327 * std::exp(-0.5 * a.value * a.value);
    return ADouble{v, g_tape->push_unary(a.index, pdf)};
}

}  // namespace

// Precio y griegas de Black-Scholes por AD en modo reverso.
//
// Devuelve delta, vega, rho, theta y la sensibilidad al dividendo con UN solo
// recorrido hacia atras. Las griegas se contrastan contra sus formulas cerradas
// en la bateria de pruebas: si la cinta estuviera mal, la discrepancia seria
// inmediata.
py::dict black_scholes_aad(double spot, double strike, double rate, double dividend, double vol,
                           double maturity, bool is_call) {
    if (spot <= 0.0 || strike <= 0.0 || vol <= 0.0 || maturity <= 0.0) {
        throw std::invalid_argument("parametros de mercado fuera de dominio");
    }
    Tape tape;
    g_tape = &tape;

    ADouble s = leaf(spot);
    ADouble k = leaf(strike);
    ADouble r = leaf(rate);
    ADouble q = leaf(dividend);
    ADouble sig = leaf(vol);
    ADouble t = leaf(maturity);

    ADouble sqrt_t = ad_sqrt(t);
    ADouble vol_sqrt_t = sig * sqrt_t;
    ADouble d1 = (ad_log(s / k) + (r - q + 0.5 * (sig * sig)) * t) / vol_sqrt_t;
    ADouble d2 = d1 - vol_sqrt_t;
    ADouble df_r = ad_exp(-1.0 * (r * t));
    ADouble df_q = ad_exp(-1.0 * (q * t));

    ADouble value;
    if (is_call) {
        value = s * df_q * ad_norm_cdf(d1) - k * df_r * ad_norm_cdf(d2);
    } else {
        ADouble zero_minus_d1 = 0.0 - d1;
        ADouble zero_minus_d2 = 0.0 - d2;
        value = k * df_r * ad_norm_cdf(zero_minus_d2) - s * df_q * ad_norm_cdf(zero_minus_d1);
    }

    const std::vector<double> adj = tape.backward(value.index);
    g_tape = nullptr;

    py::dict out;
    out["price"] = value.value;
    out["delta"] = adj[s.index];
    out["dual_delta"] = adj[k.index];  // sensibilidad al strike
    out["rho"] = adj[r.index];
    out["dividend_sensitivity"] = adj[q.index];
    out["vega"] = adj[sig.index];
    // Theta se define como la derivada respecto al *paso* del tiempo, es decir
    // el negativo de la derivada respecto al vencimiento restante.
    out["theta"] = -adj[t.index];
    out["tape_size"] = tape.size();
    return out;
}

PYBIND11_MODULE(_kernel, m) {
    m.doc() = "Nucleo nativo del motor de valuacion de derivados";

    m.def("price_gbm", &price_gbm, py::arg("normals"), py::arg("spot"), py::arg("strike"),
          py::arg("rate"), py::arg("dividend"), py::arg("vol"), py::arg("maturity"),
          py::arg("payoff_code"), py::arg("use_bridge") = true, py::arg("n_threads") = 0,
          "Monte Carlo sobre GBM. Consume las normales sin copiar y libera el GIL.");

    m.def("solve_pde", &solve_pde, py::arg("spot"), py::arg("strike"), py::arg("rate"),
          py::arg("dividend"), py::arg("vol"), py::arg("maturity"), py::arg("n_space"),
          py::arg("n_time"), py::arg("is_call") = true, py::arg("american") = false,
          py::arg("rannacher_steps") = 2, py::arg("n_stddev") = 6.0,
          py::arg("return_grid") = false,
          "Crank-Nicolson con arranque de Rannacher sobre la malla logaritmica.");

    m.def("black_scholes_aad", &black_scholes_aad, py::arg("spot"), py::arg("strike"),
          py::arg("rate"), py::arg("dividend"), py::arg("vol"), py::arg("maturity"),
          py::arg("is_call") = true,
          "Precio y todas las griegas por diferenciacion automatica en modo reverso.");

    m.attr("__has_openmp__") =
#ifdef _OPENMP
        true;
#else
        false;
#endif
}
