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

# TODO: Work through vector block motion for non-planar
# TODO: Try inclined fault
# TODO: Material parameters for each fault element
# TODO: Group other parameters into dictionary

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
v_0 = 1e-6  # when V = V0, f = f0, V is (m/s)
secs_per_year = 365 * 24 * 60 * 60
time_interval_yrs = np.linspace(0.0, 600.0, 5001)
time_interval = time_interval_yrs * secs_per_year

# Crete fault elements
ELEMENTS_FAULT = []
ELEMENT = {}
L = 10000
x1, y1, x2, y2 = bem2d.discretized_line(-L, 0, L, 0, 50)
for i in range(0, x1.size):
    ELEMENT["x1"] = x1[i]
    ELEMENT["y1"] = y1[i]
    ELEMENT["x2"] = x2[i]
    ELEMENT["y2"] = y2[i]
    ELEMENT["a"] = a
    ELEMENT["b"] = b
    ELEMENT["sigma_n"] = sigma_n
    ELEMENTS_FAULT.append(ELEMENT.copy())
ELEMENTS_FAULT = bem2d.standardize_elements(ELEMENTS_FAULT)
N_ELEMENTS = len(ELEMENTS_FAULT)
N_NODES = 3 * N_ELEMENTS

# Extract parameters for later use as arrays
ELEMENTS_FAULT_ARRAYS = {}
ELEMENTS_FAULT_ARRAYS["a"] = np.array(
    [np.tile(_["a"], 3) for _ in ELEMENTS_FAULT]
).flatten()
ELEMENTS_FAULT_ARRAYS["additional_normal_stress"] = np.array(
    [np.tile(_["sigma_n"], 3) for _ in ELEMENTS_FAULT]
).flatten()
ELEMENTS_FAULT_ARRAYS["element_normals"] = np.empty((N_ELEMENTS, 2))
ELEMENTS_FAULT_ARRAYS["element_normals"][:, 0] = np.array(
    [_["x_normal"] for _ in ELEMENTS_FAULT]
).flatten()
ELEMENTS_FAULT_ARRAYS["element_normals"][:, 1] = np.array(
    [_["y_normal"] for _ in ELEMENTS_FAULT]
).flatten()

# Calculate slip to traction partials on the fault
_, _, SLIP_TO_TRACTION = bem2d.quadratic_partials_all(
    ELEMENTS_FAULT, ELEMENTS_FAULT, mu, nu
)


def calc_state(state, velocity):
    """ State evolution law : aging law """
    return b * v_0 / Dc * (np.exp((f0 - state) / b) - (velocity / v_0))


def steady_state(velocities):  # Can I vectorize this?
    """ Steady-state state for initial condition """
    state = scipy.optimize.fsolve(calc_state, 0.5 * np.ones(N_NODES), args=(velocities))
    return state


def calc_derivatives(t, x_and_state):
    """ Derivatives to feed to ODE integrator """
    state = x_and_state[2::3]
    x = np.zeros(2 * state.size)
    x[0::2] = x_and_state[0::3]
    x[1::2] = x_and_state[1::3]

    # Current shear stress on fault (slip->traction)
    tractions = SLIP_TO_TRACTION @ x

    # Solve for the current velocity...This is the algebraic part
    current_velocity = np.empty(2 * N_NODES)
    bem2d.newton_rate_state.rate_state_solver(
        ELEMENTS_FAULT_ARRAYS["element_normals"],
        tractions,
        state,
        current_velocity,  # Modified in place and returns velocity solution
        ELEMENTS_FAULT_ARRAYS["a"],
        eta,
        v_0,
        0.0,
        ELEMENTS_FAULT_ARRAYS["additional_normal_stress"],
        TOL,
        MAXITER,
        3,
    )

    dx_dt = -current_velocity  # Is the negative sign for slip deficit convention?
    dx_dt[0::2] += Vp  # FIX TO USE Vp in the plate direction
    vel_mags = np.linalg.norm(current_velocity.reshape((-1, 2)), axis=1)
    dstate_dt = calc_state(state, vel_mags)
    derivatives = np.zeros(dx_dt.size + dstate_dt.size)
    derivatives[0::3] = dx_dt[0::2]
    derivatives[1::3] = dx_dt[1::2]
    derivatives[2::3] = dstate_dt
    return derivatives


# Set initial conditions and time integrate
initial_velocity_x = Vp / 1000.0 * np.ones(N_NODES)
initial_velocity_y = 0.0 * np.ones(N_NODES)
initial_conditions = np.zeros(3 * N_NODES)
initial_conditions[0::3] = initial_velocity_x
initial_conditions[1::3] = initial_velocity_y
initial_conditions[2::3] = steady_state(initial_velocity_x)

SOLVER = scipy.integrate.RK23(
    calc_derivatives,
    time_interval.min(),
    initial_conditions,
    time_interval.max(),
    rtol=1e-4,
    atol=1e-4,
)

SOLUTION = {}
SOLUTION["t"] = [SOLVER.t]
SOLUTION["y"] = [SOLVER.y.copy()]
while SOLVER.t < time_interval.max():
    SOLVER.step()
    print(
        f"t = {SOLVER.t / secs_per_year:05.6f}"
        + " of "
        + f"{time_interval.max() / secs_per_year:05.6f}"
        + f" ({100 * SOLVER.t / time_interval.max():.3f}"
        + "%)"
    )
    SOLUTION["t"].append(SOLVER.t)
    SOLUTION["y"].append(SOLVER.y.copy())
SOLUTION["t"] = np.array(SOLUTION["t"])
SOLUTION["y"] = np.array(SOLUTION["y"])


def plot_time_series():
    """ Plot time integrated time series for each node """
    plt.figure(figsize=(6, 9))
    y_labels = ["$u_x$ (m)", "$u_y$ (m)", "state"]
    for i, y_label in enumerate(y_labels):
        plt.subplot(3, 1, i + 1)
        plt.plot(SOLUTION["t"] / secs_per_year, SOLUTION["y"][:, i::3], linewidth=0.5)
        plt.ylabel(y_label)
    plt.xlabel("$t$ (years)")
    plt.show(block=False)


plot_time_series()


def plot_slip_profile():
    """ Plot time integrated time series for each node """
    plot_times = np.floor(np.linspace(0, SOLUTION["y"].shape[0] - 1, 50)).astype(int)
    plot_x = np.array([_["x_integration_points"] for _ in ELEMENTS_FAULT]).flatten()
    plt.figure(figsize=(6, 9))
    y_labels = ["$u_x$ (m)", "$u_y$ (m)", "state"]
    for i, y_label in enumerate(y_labels):
        plt.subplot(3, 1, i + 1)
        for j in range(plot_times.size):
            plt.plot(plot_x, SOLUTION["y"][plot_times[j], i::3], linewidth=0.5)
        plt.ylabel(y_label)
    plt.xlabel("$x$ (m)")
    plt.show(block=False)


plot_slip_profile()
