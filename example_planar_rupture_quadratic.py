import time
import numpy as np
import bem2d
import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
import scipy.optimize
import scipy.integrate
from importlib import reload

import cppimport.import_hook
import bem2d.newton_rate_state

bem2d = reload(bem2d)
plt.close("all")

# TODO: Work through vector block motion for non planar
# TODO: Try inclined fault
# TODO: Material parameters for each fault element
# TODO: Group other parameters into dictionary
# TODO: Passing of element normals (Before loop)
# TODO: Put as many functions as possible inside calc_derivatives

TOL = 1e-12
MAXITER = 50

elements_fault = []
element = {}
L = 10000
x1, y1, x2, y2 = bem2d.discretized_line(-L, 0, L, 0, 50)
for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    elements_fault.append(element.copy())
elements_fault = bem2d.standardize_elements(elements_fault)
n_elements = len(elements_fault)
n_nodes = 3 * n_elements


mu = 3e10  # Shear modulus (Pa)
nu = 0.25  # Possion's ratio
density = 2700  # rock density (kg/m^3)
cs = np.sqrt(mu / density)  # Shear wave speed (m/s)
eta = mu / (2 * cs)  # The radiation damping coefficient (kg / (m^2 * s))
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


# Calculate slip to traction partials on the fault
_, _, slip_to_traction = bem2d.quadratic_partials_all(
    elements_fault, elements_fault, mu, nu
)


def calc_frictional_stress(velocity, normal_stress, state):
    """ Rate-state friction law w/ Rice et al. (2001) regularization so that
    it is nonsingular at V = 0.  The frictional stress is equal to the
    friction coefficient * the normal stress """
    return normal_stress * a * np.arcsinh(velocity / (2 * V0) * np.exp(state / a))


def calc_state(velocity, state):
    """ State evolution law : aging law """
    return b * V0 / Dc * (np.exp((f0 - state) / b) - (velocity / V0))


def steady_state(velocities):
    """ Steady state for initial condition """
    def f(state, v):
        return calc_state(v, state)

    steady_state_state = np.zeros(n_nodes)
    for i in range(n_nodes):
        steady_state_state[i] = scipy.optimize.fsolve(f, 0.0, args=(velocities[i],))[0]
    return steady_state_state


def calc_derivatives(t, x_and_state):
    """ Derivatives to feed to ODE integrator """

    state = x_and_state[2::3]
    x = np.zeros(2 * state.size)
    x[0::2] = x_and_state[0::3]
    x[1::2] = x_and_state[1::3]

    # Current shear stress on fault (slip->traction)
    tau_qs = slip_to_traction @ x

    # Solve for the current velocity...This is the algebraic part
    def current_velocity(tau_qs, state):
        """ Solve the algebraic part of the DAE system """
        # TODO: use correct element normals!! assemble element_normals vector of shape (n_elements, 2) but do it outside this function
        current_velocities = np.empty(6 * n_elements)
        a_dofs = a * np.ones(3 * n_elements)
        additional_normal_stress = sigma_n * np.ones(3 * n_elements)
        element_normals = np.zeros((n_elements, 2))
        element_normals[:, 1] = 1.0

        bem2d.newton_rate_state.rate_state_solver(
            element_normals,
            tau_qs,
            state,
            current_velocities,
            a_dofs,
            eta,
            V0,
            0.0,
            additional_normal_stress,
            TOL,
            MAXITER,
            3,
        )
        return current_velocities
    sliding_velocity = current_velocity(tau_qs, state)

    dx_dt = -sliding_velocity
    dx_dt[0::2] += Vp  # FIX TO USE Vp in the plate direction
    # TODO: FIX TO USE VELOCITY MAGNITUDE
    vel_mags = np.linalg.norm(sliding_velocity.reshape((-1, 2)), axis=1)
    dstate_dt = calc_state(vel_mags, state)
    derivatives = np.zeros(dx_dt.size + dstate_dt.size)
    derivatives[0::3] = dx_dt[0::2]
    derivatives[1::3] = dx_dt[1::2]
    derivatives[2::3] = dstate_dt
    return derivatives


# Set initial conditions and time integrate
initial_velocity_x = Vp / 1000.0 * np.ones(n_nodes)
initial_velocity_y = 0.0 * np.ones(n_nodes)
initial_conditions = np.zeros(9 * n_elements)
initial_conditions[0::3] = initial_velocity_x
initial_conditions[1::3] = initial_velocity_y
initial_conditions[2::3] = steady_state(initial_velocity_x)


history = scipy.integrate.RK45(
    calc_derivatives,
    time_interval.min(),
    initial_conditions,
    time_interval.max(),
    rtol=1e-4,
    atol=1e-4,
)

history_t = [history.t]
history_y = [history.y.copy()]
while history.t < time_interval.max():
    history.step()
    print(
        f"t = {history.t / secs_per_year:05.6f}"
        + " of "
        + f"{time_interval.max() / secs_per_year:05.6f}"
        + f" ({100 * history.t / time_interval.max():.3f}"
        + "%)"
    )
    history_t.append(history.t)
    history_y.append(history.y.copy())
history_t = np.array(history_t)
history_y = np.array(history_y)

# Plot time integrated time series
plt.figure(figsize=(6, 9))
y_labels = ["$u_x$ (m)", "$u_y$ (m)", "state"]
for i in range(len(y_labels)):
    plt.subplot(3, 1, i + 1)
    plt.plot(history_t / secs_per_year, history_y[:, i::3], linewidth=0.5)
    plt.ylabel(y_labels[i])
plt.xlabel("$t$ (years)")
plt.show(block=False)
