# Max error as a function of N for each method
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

def explicit_euler(y0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    y[0] = y0
    for n in range(0, len(t) - 1):
        y[n + 1] = y[n] + dt * f(t[n], y[n])
    return t, y

def implicit_euler(y0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    y[0] = y0
    for n in range(0, len(t) - 1):
        # Define the equation to be solved
        func = lambda y_next: y_next - y[n] - dt * f(t[n + 1], y_next)
        # Use fsolve to solve the equation
        y[n + 1] = fsolve(func, y[n])
    return t, y

def trapezoidal_rule(y0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    y[0] = y0
    for n in range(0, len(t) - 1):
        # Define the equation to be solved
        func = lambda y_next: y_next - y[n] - dt / 2 * (f(t[n], y[n]) + f(t[n + 1], y_next))
        # Use fsolve to solve the equation
        y[n + 1] = fsolve(func, y[n])
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

def adams_bashforth(y0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    # Use RK4 to initialize the first three values
    y[0:4] = rk4(y0, t0, t0 + 4*dt, dt)[1]
    for n in range(3, len(t) - 1):
        y[n + 1] = y[n] + dt/24 * (55*f(t[n], y[n]) - 59*f(t[n - 1], y[n - 1]) + 37*f(t[n - 2], y[n - 2]) - 9*f(t[n - 3], y[n - 3]))
    return t, y

def adams_moulton(y0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    y = np.zeros_like(t)
    # Use RK4 to initialize the first three values
    y[0:4] = rk4(y0, t0, t0 + 4*dt, dt)[1]
    for n in range(3, len(t) - 1):
        # Define the equation to be solved
        func = lambda y_next: y_next - y[n] - dt/720 * (251*f(t[n+1], y_next) + 646*f(t[n], y[n]) - 264*f(t[n - 1], y[n - 1]) + 106*f(t[n - 2], y[n - 2]) - 19*f(t[n - 3], y[n - 3]))
        # Use fsolve to solve the equation
        y[n + 1] = fsolve(func, y[n])
    return t, y


# Initialize the max error list for each method
methods = [explicit_euler, implicit_euler, trapezoidal_rule, rk4, adams_bashforth, adams_moulton]
method_names = ['Explicit Euler', 'Implicit Euler', 'Trapezoidal Rule', 'RK4', 'Adams-Bashforth', 'Adams-Moulton']
max_errors = {name: [] for name in method_names}

# Define the initial conditions and time range
y0 = 1    # initial condition
t0 = 0    # start time
tf = 3    # end time

# Define the list of N values
N_values = [640, 320, 160, 80, 40]

# For each N
for N in N_values:
    dt = (tf - t0) / N

    # For each method
    for method, name in zip(methods, method_names):
        # prevent runtime warning
        if N == 40 and name == 'Explicit Euler':
            continue

        # Solve the ODE
        t, y = method(y0, t0, tf, dt)

        # Compute the max error
        max_error = np.max(np.abs(y - exact(y0, t0, tf, dt)[1]))

        # Append the max error to the list
        max_errors[name].append(max_error)

# Plot the max error as a function of N for each method
plt.figure(figsize=(8, 6))
for name in method_names:
    if name == 'Explicit Euler':
        plt.plot(np.log10(3 / np.array(N_values[0:4])), np.log10(max_errors[name]), label=name)

        # Calculate average slope
        delta_log_e = np.diff(np.log10(max_errors[name]))
        delta_log_h = np.diff(np.log10(3 / np.array(N_values[0:4])))
        slopes = delta_log_e / delta_log_h
        average_slope = np.mean(slopes)
        print(f"Average slope of {name}: ", average_slope)

        continue

    plt.plot(np.log10(3 / np.array(N_values)), np.log10(max_errors[name]), label=name)

    # Calculate average slope
    delta_log_e = np.diff(np.log10(max_errors[name]))
    delta_log_h = np.diff(np.log10(3 / np.array(N_values)))
    slopes = delta_log_e / delta_log_h
    average_slope = np.mean(slopes)
    print(f"Average slope of {name}: ", average_slope)

plt.title('Max error as a function of N for each method')
plt.xlabel('log10(3/N)')
plt.ylabel('log10(max error)')
plt.legend()
plt.grid(True)
plt.show()
