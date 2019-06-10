/*cppimport
<%
import os
setup_pybind11(cfg)
cfg['compiler_args'] += ['-std=c++14', '-O3', '-stdlib=libc++']
cfg['linker_args'] += ['-stdlib=libc++']
%>
*/

#include <utility>
#include <cmath>
#include <functional>
#include <iostream>
#include <cassert>

template <typename F, typename Fp>
std::pair<double, bool> newton(const F &f, const Fp &fp, double x0, double tol, int maxiter)
{
    for (int i = 0; i < maxiter; i++)
    {
        double y = f(x0);
        double yp = fp(x0);
        double x1 = x0 - y / yp;
        // std::cout << x0 << " " << x1 << " " << y << " " << yp << x0 - x1 << std::endl;
        if (std::fabs(x1 - x0) <= tol * std::fabs(x0))
        {
            return {x1, true};
        }
        x0 = x1;
    }
    return {x0, false};
}

double F(double V, double sigma_n, double state, double a, double V0, double C)
{
    return a * sigma_n * std::asinh(V / (2 * V0) * std::exp(state / a)) - C;
}

//https://www.wolframalpha.com/input/?i=d%5Ba*S*arcsinh(x+%2F+(2*y)+*+exp(s%2Fa))%5D%2Fdx
double dFdV(double V, double sigma_n, double state, double a, double V0)
{
    double expsa = std::exp(state / a);
    double Q = (V * expsa) / (2 * V0);
    return a * expsa * sigma_n / (2 * V0 * std::sqrt(1 + (Q * Q)));
}

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "pybind11_nparray.hpp"
#include "vec_tensor.hpp"

namespace py = pybind11;

auto newton_py(std::function<double(double)> f,
               std::function<double(double)> fp,
               double x0, double tol, int maxiter)
{
    return newton(f, fp, x0, tol, maxiter);
}

auto newton_rs(double tau_qs, double eta, double sigma_n,
               double state, double a, double V0, double C,
               double V_guess, double tol, int maxiter)
{
    auto rsf = [&](double V) {
        return tau_qs - eta * V - F(V, sigma_n, state, a, V0, C);
    };
    auto rsf_deriv = [&](double V) {
        return -eta - dFdV(V, sigma_n, state, a, V0);
    };
    auto out = newton(rsf, rsf_deriv, V_guess, tol, maxiter);
    // auto left = rsf(out.first * (1 - tol));
    // auto right = rsf(out.first * (1 + tol));
    // assert(left > out && right > out);
    return out;
}

Vec2 solve_for_dof_mag(const Vec2 &traction_vec, double state, const Vec2 &normal,
                       const std::function<double(double, double)> rs_solver)
{
    const double eps = 1e-14;
    auto normal_stress_vec = projection(traction_vec, normal);
    auto shear_traction_vec = sub(traction_vec, normal_stress_vec);

    double normal_mag = length(normal_stress_vec);
    double shear_mag = length(shear_traction_vec);
    if (shear_mag < eps)
    {
        return {0, 0};
    }
    // double shear_mag = traction_vec[0]; Uncomment to go back to x-axis only.s
    // double normal_mag = 0.0;

    double V_mag = rs_solver(shear_mag, normal_mag);

    //return {V_mag, 0.0}; Uncomment to go back to x-axis only.

    auto shear_dir = div(shear_traction_vec, shear_mag);
    return mult(shear_dir, V_mag);
}

void rate_state_solver(NPArray<double> element_normals, NPArray<double> traction,
                       NPArray<double> state, NPArray<double> velocity, NPArray<double> a,
                       double eta, double V0, double C,
                       NPArray<double> additional_normal_stress,
                       double tol, double maxiter, int basis_dim)
{
    auto *element_normals_ptr = as_ptr<Vec2>(element_normals);
    auto *state_ptr = as_ptr<double>(state);
    auto *velocity_ptr = as_ptr<Vec2>(velocity);
    auto *traction_ptr = as_ptr<Vec2>(traction);
    auto *a_ptr = as_ptr<double>(a);
    auto *normal_stress_ptr = as_ptr<double>(additional_normal_stress);

    size_t n_elements = element_normals.request().shape[0];

    // #pragma omp parallel for
    for (size_t i = 0; i < n_elements; i++)
    {
        auto normal = element_normals_ptr[i];
        for (int d = 0; d < basis_dim; d++)
        {

            size_t dof = i * basis_dim + d;
            auto traction_vec = traction_ptr[dof];
            auto state = state_ptr[dof];

            auto rs_solver_fnc = [&](double shear_mag, double normal_mag) {
                auto solve_result = newton_rs(
                    shear_mag, eta,
                    normal_mag + normal_stress_ptr[dof],
                    // normal_stress_ptr[dof], // This line ignores the time varying elastic normal stresses.
                    // and only uses the constant offset normal stress.
                    state, a_ptr[dof], V0, C, 0.0, tol, maxiter);
                assert(solve_result.second);
                return solve_result.first;
            };

            Vec2 vel = solve_for_dof_mag(traction_vec, state, normal, rs_solver_fnc);
            for (int d2 = 0; d2 < 2; d2++)
            {
                velocity_ptr[dof][d2] = vel[d2];
            }
        }
    }
}

PYBIND11_MODULE(newton_rate_state, m)
{
    m.def("newton", &newton_py);
    m.def("newton_rs", &newton_rs);
    m.def("F", F);
    m.def("dFdV", dFdV);
    m.def("rate_state_solver", rate_state_solver);
}
