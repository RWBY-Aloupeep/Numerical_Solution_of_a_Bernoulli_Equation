# Solution to dy/dt = y/(0.01+t)-y^3 for different N
# use six methods: explicit Euler’s method, implicit Euler’s method, trapezoidal rule method, RK4 method, four-step Adams-Bashforth method and four-step Adams-Moulton method
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time


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
        y[n + 1] = fsolve(func, y[n], xtol=0.1)
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


# Initial condition
y0 = 1    # initial condition
t0 = 0    # start time
tf = 3    # end time


# Solve the ODE using each method
N = 80
dt = (tf - t0) / N
t_exact_100, y_exact_100 = exact(y0, t0, tf, dt)
start_time_explicit_euler = time.time()
t_explicit_euler_100, y_explicit_euler_100 = explicit_euler(y0, t0, tf, dt)
end_time_explicit_euler = time.time()
start_time_implicit_euler = time.time()
t_implicit_euler_100, y_implicit_euler_100 = implicit_euler(y0, t0, tf, dt)
end_time_implicit_euler = time.time()
start_time_trapezoidal_rule = time.time()
t_trapezoidal_rule_100, y_trapezoidal_rule_100 = trapezoidal_rule(y0, t0, tf, dt)
end_time_trapezoidal_rule = time.time()
start_time_rk4 = time.time()
t_rk4_100, y_rk4_100 = rk4(y0, t0, tf, dt)
end_time_rk4 = time.time()
start_time_adams_bashforth = time.time()
t_adams_bashforth_100, y_adams_bashforth_100 = adams_bashforth(y0, t0, tf, dt)
end_time_adams_bashforth = time.time()
start_time_adams_moulton = time.time()
t_adams_moulton_100, y_adams_moulton_100 = adams_moulton(y0, t0, tf, dt)
end_time_adams_moulton = time.time()
N = 20
dt = (tf - t0) / N
t_exact_20, y_exact_20 = exact(y0, t0, tf, dt)
t_implicit_euler_20, y_implicit_euler_20 = implicit_euler(y0, t0, tf, dt)
t_trapezoidal_rule_20, y_trapezoidal_rule_20 = trapezoidal_rule(y0, t0, tf, dt)
N = 40
dt = (tf - t0) / N
t_exact_40, y_exact_40 = exact(y0, t0, tf, dt)
t_implicit_euler_40, y_implicit_euler_40 = implicit_euler(y0, t0, tf, dt)
t_trapezoidal_rule_40, y_trapezoidal_rule_40 = trapezoidal_rule(y0, t0, tf, dt)
t_rk4_40, y_rk4_40 = rk4(y0, t0, tf, dt)
t_adams_bashforth_40, y_adams_bashforth_40 = adams_bashforth(y0, t0, tf, dt)
t_adams_moulton_40, y_adams_moulton_40 = adams_moulton(y0, t0, tf, dt)
N = 160
dt = (tf - t0) / N
t_exact_160, y_exact_160 = exact(y0, t0, tf, dt)
t_explicit_euler_160, y_explicit_euler_160 = explicit_euler(y0, t0, tf, dt)
t_implicit_euler_160, y_implicit_euler_160 = implicit_euler(y0, t0, tf, dt)
t_trapezoidal_rule_160, y_trapezoidal_rule_160 = trapezoidal_rule(y0, t0, tf, dt)
t_rk4_160, y_rk4_160 = rk4(y0, t0, tf, dt)
t_adams_bashforth_160, y_adams_bashforth_160 = adams_bashforth(y0, t0, tf, dt)
t_adams_moulton_160, y_adams_moulton_160 = adams_moulton(y0, t0, tf, dt)


# Plot the results
fig = plt.figure(figsize=(8, 8))

# N = 100
ax1 = plt.subplot(2, 2, 1)
ax1.set_title('N = 100')
ax1.plot(t_exact_100, y_exact_100, label='Exact', color='black')
ax1.plot(t_explicit_euler_100, y_explicit_euler_100, label='Explicit Euler', color='red')
ax1.plot(t_implicit_euler_100, y_implicit_euler_100, label='Implicit Euler', color='orange')
ax1.plot(t_trapezoidal_rule_100, y_trapezoidal_rule_100, label='Trapezoidal Rule', color='green')
ax1.plot(t_rk4_100, y_rk4_100, label='RK4', color='blue')
ax1.plot(t_adams_bashforth_100, y_adams_bashforth_100, label='Adams-Bashforth', color='purple')
ax1.plot(t_adams_moulton_100, y_adams_moulton_100, label='Adams-Moulton', color='brown')
ax1.legend()
ax1.set_xlabel('t')
ax1.set_ylabel('y')
ax1.grid(True)

# N = 20
ax2 = plt.subplot(2, 2, 2)
ax2.set_title('N = 20')
ax2.plot(t_exact_20, y_exact_20, label='Exact', color='black')
ax2.plot(t_implicit_euler_20, y_implicit_euler_20, label='Implicit Euler', color='orange')
ax2.plot(t_trapezoidal_rule_20, y_trapezoidal_rule_20, label='Trapezoidal Rule', color='green')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_ylabel('y')
ax2.grid(True)

# N = 40
ax3 = plt.subplot(2, 2, 3)
ax3.set_title('N = 40')
ax3.plot(t_exact_40, y_exact_40, label='Exact', color='black')
ax3.plot(t_implicit_euler_40, y_implicit_euler_40, label='Implicit Euler', color='orange')
ax3.plot(t_trapezoidal_rule_40, y_trapezoidal_rule_40, label='Trapezoidal Rule', color='green')
ax3.plot(t_rk4_40, y_rk4_40, label='RK4', color='blue')
ax3.plot(t_adams_bashforth_40, y_adams_bashforth_40, label='Adams-Bashforth', color='purple')
ax3.plot(t_adams_moulton_40, y_adams_moulton_40, label='Adams-Moulton', color='brown')
ax3.legend()
ax3.set_xlabel('t')
ax3.set_ylabel('y')
ax3.grid(True)

# N = 160
ax4 = plt.subplot(2, 2, 4)
ax4.set_title('N = 160')
ax4.plot(t_exact_160, y_exact_160, label='Exact', color='black')
ax4.plot(t_explicit_euler_160, y_explicit_euler_160, label='Explicit Euler', color='red')
ax4.plot(t_implicit_euler_160, y_implicit_euler_160, label='Implicit Euler', color='orange')
ax4.plot(t_trapezoidal_rule_160, y_trapezoidal_rule_160, label='Trapezoidal Rule', color='green')
ax4.plot(t_rk4_160, y_rk4_160, label='RK4', color='blue')
ax4.plot(t_adams_bashforth_160, y_adams_bashforth_160, label='Adams-Bashforth', color='purple')
ax4.plot(t_adams_moulton_160, y_adams_moulton_160, label='Adams-Moulton', color='brown')
ax4.legend()
ax4.set_xlabel('t')
ax4.set_ylabel('y')
ax4.grid(True)

fig.suptitle('Solution to dy/dt = y/(0.01+t)-y^3 for different N', fontsize=16)
plt.tight_layout()
plt.show()


# calculate the time
elapsed_time_explicit_euler = end_time_explicit_euler - start_time_explicit_euler
elapsed_time_implicit_euler = end_time_implicit_euler - start_time_implicit_euler
elapsed_time_trapezoidal_rule = end_time_trapezoidal_rule - start_time_trapezoidal_rule
elapsed_time_rk4 = end_time_rk4 - start_time_rk4
elapsed_time_adams_bashforth = end_time_adams_bashforth - start_time_adams_bashforth
elapsed_time_adams_moulton = end_time_adams_moulton - start_time_adams_moulton
print(f"The code of explicit Euler's method for N = 100 took {elapsed_time_explicit_euler} seconds to run.")
print(f"The code of implicit Euler's method for N = 100 took {elapsed_time_implicit_euler} seconds to run.")
print(f"The code of trapezoidal rule method for N = 100 took {elapsed_time_trapezoidal_rule} seconds to run.")
print(f"The code of RK4 method for N = 100 took {elapsed_time_rk4} seconds to run.")
print(f"The code of four-step Adams-Bashforth method for N = 100 took {elapsed_time_adams_bashforth} seconds to run.")
print(f"The code of four-step Adams-Moulton method for N = 100 took {elapsed_time_adams_moulton} seconds to run.")
