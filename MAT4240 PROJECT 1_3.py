# Max error as a function of tolerance for three implicit methods
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def f(t, y):
    return y / (t + 0.01) - y ** 3


def exact(y0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    y[0] = y0
    y = (0.01 + t) / np.sqrt(0.0001 - 2 / 3 * 0.000001 + 2 / 3 * (0.01 + t) ** 3)
    return t, y

def rk4(y0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    y[0] = y0
    for n in range(0, len(t) - 1):
        k1 = dt * f(t[n], y[n])
        k2 = dt * f(t[n] + dt/2, y[n] + k1/2)
        k3 = dt * f(t[n] + dt/2, y[n] + k2/2)
        k4 = dt * f(t[n] + dt, y[n] + k3)
        y[n + 1] = y[n] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

def implicit_euler(y0, t0, tf, dt, xtol):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    y[0] = y0
    for n in range(0, len(t) - 1):
        # Define the equation to be solved
        func = lambda y_next: y_next - y[n] - dt * f(t[n + 1], y_next)
        # Use fsolve to solve the equation
        y[n + 1] = fsolve(func, y[n], xtol=xtol)
    return t, y

def trapezoidal_rule(y0, t0, tf, dt, xtol):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    y[0] = y0
    for n in range(0, len(t) - 1):
        # Define the equation to be solved
        func = lambda y_next: y_next - y[n] - dt / 2 * (f(t[n], y[n]) + f(t[n + 1], y_next))
        # Use fsolve to solve the equation
        y[n + 1] = fsolve(func, y[n], xtol=xtol)
    return t, y

def adams_moulton(y0, t0, tf, dt, xtol):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    # Use RK4 to initialize the first three values
    y[0:4] = rk4(y0, t0, t0 + 4*dt, dt)[1]
    for n in range(3, len(t) - 1):
        # Define the equation to be solved
        func = lambda y_next: y_next - y[n] - dt/720 * (251*f(t[n+1], y_next) + 646*f(t[n], y[n]) - 264*f(t[n - 1], y[n - 1]) + 106*f(t[n - 2], y[n - 2]) - 19*f(t[n - 3], y[n - 3]))
        # Use fsolve to solve the equation
        y[n + 1] = fsolve(func, y[n], xtol=xtol)
    return t, y


# Initialize the max error list for each method
methods = [implicit_euler, trapezoidal_rule, adams_moulton]
method_names = ['Implicit Euler', 'Trapezoidal Rule', 'Adams-Moulton']

# Define the initial conditions and time range
y0 = 1    # initial condition
t0 = 0    # start time
tf = 3    # end time

N_values = [80, 160, 320, 640]

# Define the list of tol values
tolerances = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]

# Plot the max error as a function of tol for each method
fig = plt.figure(figsize=(8, 6))

# For each tol
for N in N_values:
    max_errors = {name: [] for name in method_names}
    dt = (tf - t0) / N

    for tol in tolerances:

        for method, name in zip(methods, method_names):

            t, y = method(y0, t0, tf, dt, tol)
            max_error = np.max(np.abs(y - exact(y0, t0, tf, dt)[1]))
            max_errors[name].append(max_error)

    ax = plt.subplot(2, 2, int(np.log2(N / 80)) + 1)
    ax.set_title(f'N = {N}')
    for name in method_names:
        plt.plot(np.log10(np.array(tolerances)), np.log10(max_errors[name]), label=name)
    ax.legend(loc='upper left')
    ax.set_xlabel('log10(tol)')
    ax.set_ylabel('log10(max error)')
    ax.grid(True)

fig.suptitle('Max error as a function of tol for three implicit methods', fontsize=16)
plt.tight_layout()
plt.show()
