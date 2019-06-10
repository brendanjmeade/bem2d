import time
import numpy as np
import bem2d
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import ode, odeint, RK45
from importlib import reload

import cppimport.import_hook
import bem2d.newton_rate_state

bem2d = reload(bem2d)
plt.close("all")

""" Taken from Ben's 1d example:
    http://tbenthompson.com/post/block_slider/ """

# TODO: Save information from rupture problem as .pkl/.npz
# TODO: Try rupture problem with variations in a-b.  Do I have to pass elements_* dict to do this?


elements_fault = []
element = {}
L = 10000
mu = 3e10
nu = 0.25
x1, y1, x2, y2 = bem2d.discretized_line(-L, 0, L, 0, 50)

for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements_fault.append(element.copy())
elements_fault = bem2d.standardize_elements(elements_fault)
n_elements = len(elements_fault)

# Constant slip partial derivatives
slip_to_displacement, slip_to_traction = bem2d.constant_linear_partials(
    elements_fault, elements_fault, "slip", mu, nu
)

# Quadratic slip partial derivative
slip_to_displacement_quadratic, slip_to_stress_quadratic, slip_to_traction_quadratic = bem2d.quadratic_partials_all(
    elements_fault, elements_fault, mu, nu
)

sm = 3e10  # Shear modulus (Pa)
density = 2700  # rock density (kg/m^3)
cs = np.sqrt(sm / density)  # Shear wave speed (m/s)
eta = sm / (2 * cs)  # The radiation damping coefficient (kg / (m^2 * s))
Vp = 1e-9  # Rate of plate motion
sigma_n = 50e6  # Normal stress (Pa)
a = 0.015  # direct velocity strengthening effect
b = 0.02  # state-based velocity weakening effect
Dc = 0.05  # state evolution length scale (m)
f0 = 0.6  # baseline coefficient of friction
V0 = 1e-6  # when V = V0, f = f0, V is (m/s)
secs_per_year = 365 * 24 * 60 * 60
time_interval_yrs = np.linspace(0.0, 600.0, 5001)
time_interval = time_interval_yrs * secs_per_year

# Initially, the slider is moving at 1/1000th the plate rate.
initial_velocity = np.zeros(2 * n_elements)
initial_velocity[0::2] = Vp / 1000.0
initial_velocity_quadratic = np.zeros(6 * n_elements)
initial_velocity_quadratic[0::2] = Vp / 1000.0


def calc_frictional_stress(velocity, normal_stress, state):
    """ Rate-state friction law w/ Rice et al 2001 regularization so that
    it is nonsingular at V = 0.  The frictional stress is equal to the
    friction coefficient * the normal stress """
    friction = a * np.arcsinh(velocity / (2 * V0) * np.exp(state / a))
    frictional_stress = friction * normal_stress
    return frictional_stress


def calc_frictional_stress(velocity, normal_stress, state):
    """ Rate-state friction law w/ Rice et al 2001 regularization so that
    it is nonsingular at V = 0.  The frictional stress is equal to the
    friction coefficient * the normal stress """
    friction = a * np.arcsinh(velocity / (2 * V0) * np.exp(state / a))
    frictional_stress = friction * normal_stress
    return frictional_stress

def calc_state(velocity, state):
    """ State evolution law - aging law """
    return (b * V0 / Dc) * (np.exp((f0 - state) / b) - (velocity / V0))


a_dofs = a * np.ones(n_elements)
additional_normal_stress = sigma_n * np.ones(n_elements)
element_normals = np.zeros((n_elements, 2))
element_normals[:, 1] = 1.0

def current_velocity(tau_qs, state, V_old):
    current_velocities2 = np.empty(n_elements * 2)
    tol = 1e-12
    maxiter = 50
    bem2d.newton_rate_state.rate_state_solver(
        element_normals, tau_qs, state, current_velocities2,
        a_dofs, eta, V0, 0.0, additional_normal_stress,
        tol, maxiter, 1
    )
    return current_velocities2


    """ Solve the algebraic part of the DAE system """
    def f(V, tau_local, normal_stress, state_local):
        return (
            tau_local - eta * V - calc_frictional_stress(V, normal_stress, state_local)
        )

    def fp(V, tau_local, normal_stress, state_local):
        expsa = np.exp(state_local / a);
        Q = (V * expsa) / (2 * V0);
        return -eta - a * expsa * normal_stress / (2 * V0 * np.sqrt(1 + (Q * Q)));

    # For each element do the f(V) solve
    # current_velocities = np.zeros(2 * n_elements)
    # velocity_mag = fsolve(
    #     f, V_old[0::2], args=(tau_qs[0::2], sigma_n * np.ones(n_elements), state)
    # )
    # ONLY FOR FLAT GEOMETERY with y = 0 on all elements
    # current_velocities[0::2] = velocity_mag
    # current_velocities[1::2] = 0

    # # For each element do the f(V) solve
    current_velocities = np.zeros(2 * n_elements)
    for i in range(0, n_elements):
        shear_stress = tau_qs[2 * i]
        normal_stress = sigma_n
        velocity_mag = fsolve(
            f, V_old[2 * i], args=(shear_stress, normal_stress, state[i]),
            fprime = fp,
            full_output = True,
            xtol = 1e-13
        )
        # ONLY FOR FLAT GEOMETERY with y = 0 on all elements
        current_velocities[2 * i] = velocity_mag[0]
        current_velocities[2 * i + 1] = 0
        print("Python: ", shear_stress, velocity_mag, eta, sigma_n, state[i], a, V0, 0.0)

    np.testing.assert_almost_equal(current_velocities, current_velocities2.flatten())
    return current_velocities


def current_velocity_quadratic(tau_qs, state, V_old):
    """ Solve the algebraic part of the DAE system """
    #TODO: use correct element normals!! assemble element_normals vector of shape (n_elements, 2) but do it outside this function

    current_velocities = np.empty(6 * n_elements)

    a_dofs = a * np.ones(3 * n_elements)
    additional_normal_stress = sigma_n * np.ones(3 * n_elements)
    element_normals = np.zeros((n_elements, 2))
    element_normals[:, 1] = 1.0

    tol = 1e-12
    maxiter = 50
    bem2d.newton_rate_state.rate_state_solver(
        element_normals, tau_qs, state, current_velocities,
        a_dofs, eta, V0, 0.0, additional_normal_stress,
        tol, maxiter, 3
    )
    return current_velocities
    # def f(V, tau_local, normal_stress, state_local):
    #     return (
    #         tau_local - eta * V - calc_frictional_stress(V, normal_stress, state_local)
    #     )

    # # For each node do the f(V) solve
    # current_velocities = np.zeros(6 * n_elements)
    # for i in range(0, 3 * n_elements):
    #     velocity_mag = fsolve(f, V_old[2 * i], args=(tau_qs[2 * i], sigma_n, state[i]))[
    #         0
    #     ]
    #     current_velocities[2 * i] = velocity_mag
    #     current_velocities[2 * i + 1] = 0  # Assuming x-direction only
    # return current_velocities


def steady_state(velocities):
    """ Steady state...state """
    # steady_state_state = np.zeros(n_elements)

    def f(state, v):
        return calc_state(v, state)

    x_vels = velocities[0::2]
    steady_state_state = fsolve(f, np.zeros(x_vels.shape), args=(x_vels,))[0]

    # for i in range(0, n_elements):
    #     # TODO: FIX FOR NON XAXIS FAULT, USE VELOCITY MAGNITUDE
    #     steady_state_state[i] = fsolve(f, 0.0, args=(velocities[2 * i],))[0]
    return steady_state_state
state_0 = steady_state(initial_velocity)


def steady_state_quadratic(velocities):
    """ Steady state...state """
    steady_state_state = np.zeros(3 * n_elements)

    def f(state, v):
        return calc_state(v, state)

    for i in range(3 * n_elements):
        # TODO: FIX FOR NON XAXIS FAULT, USE VELOCITY MAGNITUDE
        steady_state_state[i] = fsolve(f, 0.0, args=(velocities[2 * i],))[0]
    return steady_state_state


state_0_quadratic = steady_state_quadratic(initial_velocity_quadratic)


def calc_derivatives(x_and_state, t):
    """ Derivatives to feed to ODE integrator """
    calc_derivatives.idx += 1
    if calc_derivatives.idx % 100 == 0:
        print("time =", t / secs_per_year)

    ux = x_and_state[0::3]
    uy = x_and_state[1::3]
    state = x_and_state[2::3]
    x = np.zeros(ux.size + uy.size)
    x[0::2] = ux
    x[1::2] = uy

    # Current shear stress on fault (slip->traction)
    tau_qs = slip_to_traction @ x

    # Solve for the current velocity...This is the algebraic part
    sliding_velocity = current_velocity(
        tau_qs, state, calc_derivatives.sliding_velocity_old
    )

    # Store the velocity to use it next time for warm-start the velocity solver
    calc_derivatives.sliding_velocity_old = sliding_velocity
    dx_dt = -sliding_velocity
    dx_dt[0::2] += Vp  # FIX TO USE Vp in the plate direction
    # TODO: FIX TO USE VELOCITY MAGNITUDE
    dstate_dt = calc_state(sliding_velocity[0::2], state)
    derivatives = np.zeros(dx_dt.size + dstate_dt.size)
    derivatives[0::3] = dx_dt[0::2]
    derivatives[1::3] = dx_dt[1::2]
    derivatives[2::3] = dstate_dt
    return derivatives


# def calc_derivatives_ode(t, x_and_state):
#     """ Derivatives to feed to ODE integrator """
#     calc_derivatives.idx += 1
#     if calc_derivatives.idx % 100 == 0:
#         print("time =", t / secs_per_year)

#     ux = x_and_state[0::3]
#     uy = x_and_state[1::3]
#     state = x_and_state[2::3]
#     x = np.zeros(ux.size + uy.size)
#     x[0::2] = ux
#     x[1::2] = uy

#     # Current shear stress on fault (slip->traction)
#     tau_qs = slip_to_traction @ x

#     # Solve for the current velocity...This is the algebraic part
#     sliding_velocity = current_velocity(
#         tau_qs, state, calc_derivatives.sliding_velocity_old
#     )

#     # Store the velocity to use it next time for warm-start the velocity solver
#     calc_derivatives.sliding_velocity_old = sliding_velocity
#     dx_dt = -sliding_velocity
#     dx_dt[0::2] += Vp  # FIX TO USE Vp in the plate direction
#     # TODO: FIX TO USE VELOCITY MAGNITUDE
#     dstate_dt = calc_state(sliding_velocity[0::2], state)
#     derivatives = np.zeros(dx_dt.size + dstate_dt.size)
#     derivatives[0::3] = dx_dt[0::2]
#     derivatives[1::3] = dx_dt[1::2]
#     derivatives[2::3] = dstate_dt
#     return derivatives

calc_derivatives.idx = 0
calc_derivatives.sliding_velocity_old = initial_velocity

def calc_derivatives_quadratic(x_and_state, t):
    """ Derivatives to feed to ODE integrator """
    calc_derivatives_quadratic.idx += 1
    if calc_derivatives_quadratic.idx % 100 == 0:
        print("time =", t / secs_per_year)

    ux = x_and_state[0::3]
    uy = x_and_state[1::3]
    state = x_and_state[2::3]
    x = np.zeros(ux.size + uy.size)
    x[0::2] = ux
    x[1::2] = uy

    # Current shear stress on fault (slip->traction)
    tau_qs = slip_to_traction_quadratic @ x

    # Solve for the current velocity...This is the algebraic part
    sliding_velocity_quadratic = current_velocity_quadratic(
        tau_qs, state, calc_derivatives_quadratic.sliding_velocity_old
    )

    # Store the velocity to use it next time for warm-start the velocity solver
    calc_derivatives.sliding_velocity_old = sliding_velocity_quadratic

    dx_dt = -sliding_velocity_quadratic
    dx_dt[0::2] += Vp  # FIX TO USE Vp in the plate direction
    # TODO: FIX TO USE VELOCITY MAGNITUDE
    vel_mags = np.linalg.norm(sliding_velocity_quadratic.reshape((-1,2)), axis = 1)
    dstate_dt = calc_state(vel_mags, state)
    derivatives = np.zeros(dx_dt.size + dstate_dt.size)
    derivatives[0::3] = dx_dt[0::2]
    derivatives[1::3] = dx_dt[1::2]
    derivatives[2::3] = dstate_dt
    return derivatives


calc_derivatives_quadratic.idx = 0
calc_derivatives_quadratic.sliding_velocity_old = initial_velocity_quadratic


# Set initial conditions
displacement_fault = np.zeros(2 * n_elements)
state_fault = state_0 * np.ones(n_elements)
initial_conditions = np.zeros(3 * n_elements)
initial_conditions[0::3] = displacement_fault[0::2]
initial_conditions[1::3] = displacement_fault[1::2]
initial_conditions[2::3] = state_fault

displacement_fault_quadratic = np.zeros(6 * n_elements)
state_fault_quadratic = state_0_quadratic * np.ones(3 * n_elements)
initial_conditions_quadratic = np.zeros(9 * n_elements)
initial_conditions_quadratic[0::3] = displacement_fault_quadratic[0::2]
initial_conditions_quadratic[1::3] = displacement_fault_quadratic[1::2]
initial_conditions_quadratic[2::3] = state_fault_quadratic

# Time the derivayive calculation
def benchmark_derivative_evaluation():
    x_and_state = np.random.rand(3 * n_elements)
    t = np.random.rand(1)
    _ = calc_derivatives(t, x_and_state)

    n_tests = 100

    start_time = time.time()
    for i in range(n_tests):
        x_and_state = np.random.rand(3 * n_elements)
        t = np.random.rand(1)
        _ = calc_derivatives(t, x_and_state)
    end_time = time.time()
    print("(constant derivative evaluation)")
    print("--- %s seconds ---" % (end_time - start_time))

    start_time = time.time()
    for i in range(n_tests):
        slip = np.random.rand(2 * n_elements)
        _ = slip_to_traction * slip
    end_time = time.time()
    print("constant (matrix vector multiply)")
    print("--- %s seconds ---" % (end_time - start_time))

    start_time = time.time()
    for i in range(n_tests):
        x_and_state = np.random.rand(9 * n_elements)
        t = np.random.rand(1)
        _ = calc_derivatives_quadratic(t, x_and_state)
    end_time = time.time()
    print("quadratic (derivative evaluation)")
    print("--- %s seconds ---" % (end_time - start_time))

    start_time = time.time()
    for i in range(n_tests):
        slip = np.random.rand(6 * n_elements)
        _ = slip_to_traction_quadratic * slip
    end_time = time.time()
    print("quadratic (matrix vector multiply)")
    print("--- %s seconds ---" % (end_time - start_time))
# benchmark_derivative_evaluation()


# # Integrate to build time series
# history = odeint(
#     calc_derivatives,
#     initial_conditions,
#     time_interval,
#     rtol=1e-4,
#     atol=1e-4,
#     mxstep=5000,
#     printmessg=True,
# )


# Integrate to build time series
# history_RK45 = RK45(
#     calc_derivatives,
#     time_interval.min(),
#     initial_conditions,
#     time_interval.max(),
#     rtol=1e-4,
#     atol=1e-4,
#     mxstep=5000,
# )

history_RK45 = RK45(
    lambda t, x_and_state: calc_derivatives_quadratic(x_and_state, t),
    0,
    initial_conditions_quadratic,
    1e100,
    rtol=1e-4,
    atol=1e-4
)

# while history_RK45.t < time_interval.max():
history_RK45_t = []
history_RK45_y = []
for i in range(20000):
    if history_RK45.t > 800 * secs_per_year:
        break
    history_RK45.step()
    history_RK45_t.append(history_RK45.t)
    history_RK45_y.append(history_RK45.y.copy())
    
history_RK45_t = np.array(history_RK45_t)
history_RK45_y = np.array(history_RK45_y)

# Plot time series
plt.figure(figsize=(6, 9))
plt.subplot(3, 1, 1)
for i in range(n_elements):
    plt.plot(history_RK45_t, history_RK45_y[:, 3 * i], label=str(i), linewidth=0.5)
plt.xlabel("years")
plt.ylabel("$u_x$")

plt.subplot(3, 1, 2)
for i in range(n_elements):
    plt.plot(history_RK45_t, history_RK45_y[:, (3 * i) + 1], label=str(i), linewidth=0.5)
plt.xlabel("years")
plt.ylabel("$u_y$")

plt.subplot(3, 1, 3)
for i in range(n_elements):
    plt.plot(history_RK45_t, history_RK45_y[:, (3 * i) + 2], label=str(i), linewidth=0.5)
plt.xlabel("years")
plt.ylabel("state")
plt.show(block=False)



# # Quadratic integrations
# history = odeint(
#     calc_derivatives_quadratic,
#     initial_conditions_quadratic,
#     time_interval,
#     rtol=1e-4,
#     atol=1e-4,
#     mxstep=5000,
#     printmessg=True,
# )

# print("finished integration")
# # Plot time series
# plt.figure(figsize=(6, 9))
# plt.subplot(3, 1, 1)
# for i in range(3 * n_elements):
#     plt.plot(time_interval_yrs, history[:, (3 * i)], label=str(i), linewidth=0.5)
# plt.xlabel("years")
# plt.ylabel("$u_x$")

# plt.subplot(3, 1, 2)
# for i in range(3 * n_elements):
#     plt.plot(time_interval_yrs, history[:, (3 * i) + 1], label=str(i), linewidth=0.5)
# plt.xlabel("years")
# plt.ylabel("$u_y$")

# plt.subplot(3, 1, 3)
# for i in range(3 * n_elements):
#     plt.plot(time_interval_yrs, history[:, (3 * i) + 2], label=str(i), linewidth=0.5)
# plt.xlabel("years")
# plt.ylabel("state")
# plt.show(block=False)


# # Plot resolved tracions
# def plot_tractions():
#     plt.figure()
#     index = np.arange(0, 3 * n_elements)
#     slip_quadratic = np.zeros(6 * n_elements)
#     slip_quadratic[0::2] = 1
#     tractions_quadratic = slip_to_traction_quadratic @ slip_quadratic
#     slip_constant = np.zeros(2 * n_elements)
#     slip_constant[0::2] = 1
#     tractions_constant = slip_to_traction @ slip_constant
#     plt.plot(
#         index[2::3],
#         tractions_constant[0::2],
#         "--r",
#         label="$t_x$ (constant)",
#         linewidth=0.5,
#     )
#     plt.plot(
#         index[2::3],
#         tractions_constant[1::2],
#         "--k",
#         label="$t_y$ (constant)",
#         linewidth=0.5,
#     )
#     plt.plot(
#         index, tractions_quadratic[0::2], "-r", label="$t_x$ (quadratic)", linewidth=0.5
#     )
#     plt.plot(
#         index, tractions_quadratic[1::2], "-k", label="$t_y$ (quadratic)", linewidth=0.5
#     )
#     plt.xlabel("indexed position along fault")
#     plt.ylabel("traction (Pa)")
#     plt.title("tractions (strike-slip motion only)")
#     plt.legend()
#     plt.show(block=False)
