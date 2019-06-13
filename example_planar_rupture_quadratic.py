from importlib import reload
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate

import cppimport.import_hook
import bem2d
import bem2d.newton_rate_state


bem2d = reload(bem2d)
plt.close("all")

# TODO: Work through vector block motion for non planar
# TODO: Try inclined fault
# TODO: Material parameters for each fault element
# TODO: Group other parameters into dictionary
# TODO: Passing of element normals (Before loop)

# DAE solver parameters
TOL = 1e-12
MAXITER = 50

# Material properties
mu = 3e10  # Shear modulus (Pa)
nu = 0.25  # Possion's ratio
density = 2700  # rock density (kg/m^3)
cs = np.sqrt(mu / density)  # Shear wave speed (m/s)
eta = mu / (2 * cs)  # The radiation damping coefficient (kg / (m^2 * s))
Dc = 0.05  # state evolution length scale (m)
f0 = 0.6  # baseline coefficient of friction

# Unique to each fault segment
Vp = 1e-9  # Rate of plate motion
sigma_n = 50e6  # Normal stress (Pa)
a = 0.015  # direct velocity strengthening effect
b = 0.02  # state-based velocity weakening effect
V0 = 1e-6  # when V = V0, f = f0, V is (m/s)
secs_per_year = 365 * 24 * 60 * 60
time_interval_yrs = np.linspace(0.0, 600.0, 5001)
time_interval = time_interval_yrs * secs_per_year

# Crete fault elements
elements_fault = []
element = {}
L = 10000
x1, y1, x2, y2 = bem2d.discretized_line(-L, 0, L, 0, 50)
for i in range(0, x1.size):
    element["x1"] = x1[i]
    element["y1"] = y1[i]
    element["x2"] = x2[i]
    element["y2"] = y2[i]
    element["a"] = a
    element["sigma_n"] = sigma_n
    elements_fault.append(element.copy())
elements_fault = bem2d.standardize_elements(elements_fault)
n_elements = len(elements_fault)
n_nodes = 3 * n_elements

# Extract parameters for later use as arrays
ELEMENTS_FAULT_ARRAYS = {}
ELEMENTS_FAULT_ARRAYS["a"] = np.array(
    [np.tile(_["a"], 3) for _ in elements_fault]
).flatten()
ELEMENTS_FAULT_ARRAYS["additional_normal_stress"] = np.array(
    [np.tile(_["sigma_n"], 3) for _ in elements_fault]
).flatten()
ELEMENTS_FAULT_ARRAYS["element_normals"] = np.empty((n_elements, 2))
ELEMENTS_FAULT_ARRAYS["element_normals"][:, 0] = np.array(
    [_["x_normal"] for _ in elements_fault]
).flatten()
ELEMENTS_FAULT_ARRAYS["element_normals"][:, 1] = np.array(
    [_["y_normal"] for _ in elements_fault]
).flatten()

# Calculate slip to traction partials on the fault
_, _, slip_to_traction = bem2d.quadratic_partials_all(
    elements_fault, elements_fault, mu, nu
)


def calc_state(velocity, state):
    """ State evolution law : aging law """
    return b * V0 / Dc * (np.exp((f0 - state) / b) - (velocity / V0))


def steady_state(velocities):
    """ Steady state for initial condition """

    def f(state, v):  # Is this function neccsary?
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
    tractions = slip_to_traction @ x

    # Solve for the current velocity...This is the algebraic part
    current_velocity = np.empty(2 * n_nodes)
    bem2d.newton_rate_state.rate_state_solver(
        ELEMENTS_FAULT_ARRAYS["element_normals"],
        tractions,
        state,
        current_velocity,  # Modified in place
        ELEMENTS_FAULT_ARRAYS["a"],
        eta,
        V0,
        0.0,
        ELEMENTS_FAULT_ARRAYS["additional_normal_stress"],
        TOL,
        MAXITER,
        3,
    )

    dx_dt = -current_velocity  # Is the negative sign for slip deficit convention?
    dx_dt[0::2] += Vp  # FIX TO USE Vp in the plate direction
    vel_mags = np.linalg.norm(current_velocity.reshape((-1, 2)), axis=1)
    dstate_dt = calc_state(vel_mags, state)
    derivatives = np.zeros(dx_dt.size + dstate_dt.size)
    derivatives[0::3] = dx_dt[0::2]
    derivatives[1::3] = dx_dt[1::2]
    derivatives[2::3] = dstate_dt
    return derivatives


# Set initial conditions and time integrate
initial_velocity_x = Vp / 1000.0 * np.ones(n_nodes)
initial_velocity_y = 0.0 * np.ones(n_nodes)
initial_conditions = np.zeros(3 * n_nodes)
initial_conditions[0::3] = initial_velocity_x
initial_conditions[1::3] = initial_velocity_y
initial_conditions[2::3] = steady_state(initial_velocity_x)

solver = scipy.integrate.RK23(
    calc_derivatives,
    time_interval.min(),
    initial_conditions,
    time_interval.max(),
    rtol=1e-4,
    atol=1e-4,
)

solution = {}
solution["t"] = [solver.t]
solution["y"] = [solver.y.copy()]
while solver.t < time_interval.max():
    solver.step()
    print(
        f"t = {solver.t / secs_per_year:05.6f}"
        + " of "
        + f"{time_interval.max() / secs_per_year:05.6f}"
        + f" ({100 * solver.t / time_interval.max():.3f}"
        + "%)"
    )
    solution["t"].append(solver.t)
    solution["y"].append(solver.y.copy())
solution["t"] = np.array(solution["t"])
solution["y"] = np.array(solution["y"])

# Plot time integrated time series
plt.figure(figsize=(6, 9))
y_labels = ["$u_x$ (m)", "$u_y$ (m)", "state"]
for i, y_label in enumerate(y_labels):
    plt.subplot(3, 1, i + 1)
    plt.plot(solution["t"] / secs_per_year, solution["y"][:, i::3], linewidth=0.5)
    plt.ylabel(y_label)
plt.xlabel("$t$ (years)")
plt.show(block=False)
