import numpy as np
import bem2d
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import ode, odeint


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

# Build partial derivative matrices for Ben's thrust fault problem
slip_to_displacement, slip_to_traction = bem2d.constant_linear_partials(
    elements_fault, elements_fault, "slip", mu, nu
)
traction_to_displacement, traction_to_traction = bem2d.constant_linear_partials(
    elements_fault, elements_fault, "traction", mu, nu
)

plt.matshow(slip_to_displacement)
plt.colorbar()
plt.title("slip to displacement")

plt.matshow(slip_to_traction)
plt.colorbar()
plt.title("slip to traction")

plt.matshow(traction_to_displacement)
plt.colorbar()
plt.title("traction to displacement")

plt.matshow(traction_to_traction)
plt.colorbar()
plt.title("traction to traction")
plt.show(block=False)

n_elements = len(elements_fault)
sm = 3e10  # Shear modulus (Pa)
density = 2700  # rock density (kg/m^3)
cs = np.sqrt(sm / density)  # Shear wave speed (m/s)
eta = sm / (2 * cs)  # The radiation damping coefficient (kg / (m^2 * s))
Vp = 1e-9  # Rate of plate motion
sigma_n = 50e6  # Normal stress (Pa)
a = 0.015  # direct velocity strengthening effect
b = 0.02  # state-based velocity weakening effect
Dc = 0.1  # state evolution length scale (m)
f0 = 0.6  # baseline coefficient of friction
V0 = 1e-6  # when V = V0, f = f0, V is (m/s)
secs_per_year = 365 * 24 * 60 * 60
time_interval_yrs = np.linspace(0.0, 1000.0, 5001)
time_interval = time_interval_yrs * secs_per_year

initial_velocity = np.zeros(2 * n_elements)
initial_velocity[0::2] = (
    Vp / 1000.0
)  # Initially, the slider is moving at 1/1000th the plate rate.


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


def current_velocity(tau_qs, state, V_old):
    """ Solve the algebraic part of the DAE system """

    def f(V, tau_local, normal_stress, state_local):
        return (
            tau_local - eta * V - calc_frictional_stress(V, normal_stress, state_local)
        )

    # For each element do the f(V) solve
    current_velocities = np.zeros(2 * n_elements)
    for i in range(0, n_elements):
        shear_stress = tau_qs[2 * i]
        # shear_dir = ... # Come back to this later for non x-axis geometry
        normal_stress = sigma_n
        velocity_mag = fsolve(
            f, V_old[2 * i], args=(shear_stress, normal_stress, state[i])
        )[0]
        # ONLY FOR FLAT GEOMETERY with y = 0 on all elements
        current_velocities[2 * i] = velocity_mag
        current_velocities[2 * i + 1] = 0
    return current_velocities


def steady_state(velocities):
    """ Steady state...state """
    steady_state_state = np.zeros(n_elements)

    def f(state, v):
        return calc_state(v, state)

    for i in range(0, n_elements):
        # TODO: FIX FOR NON XAXIS FAULT, USE VELOCITY MAGNITUDE
        steady_state_state[i] = fsolve(f, 0.0, args=(velocities[2 * i],))[0]
    return steady_state_state


state_0 = steady_state(initial_velocity)


def calc_derivatives(x_and_state, t):
    """ Derivatives to feed to ODE integrator """
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


calc_derivatives.sliding_velocity_old = initial_velocity

displacement_fault = np.zeros(2 * n_elements)
state_fault = state_0 * np.ones(n_elements)
initial_conditions = np.zeros(3 * n_elements)
initial_conditions[0::3] = displacement_fault[0::2]
initial_conditions[1::3] = displacement_fault[1::2]
initial_conditions[2::3] = state_fault
print(initial_conditions)
# history = odeint(calc_derivatives, initial_conditions, time_interval, rtol=1e-12, atol=1e-12, mxstep=5000)
history = odeint(
    calc_derivatives,
    initial_conditions,
    time_interval,
    rtol=1e-13,
    atol=1e-13,
    mxstep=5000,
    printmessg=True,
)
plt.close("all")
plt.figure()
for i in range(n_elements):
    plt.plot(history[:, 3 * i], label=str(i), linewidth=0.5)
    # plt.figure()
    # plt.plot(history[:,2])
plt.legend()
plt.show(block=False)

# Save as .npz file:
# TODO: add a UUID to this
# np.savez("model_run_huge_even_linear.npz", history, time_interval)