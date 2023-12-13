import numpy as np
from numpy import exp, log


def model(u, g):
    return u[0] * np.exp(-u[0]**2 - u[1]**2) + (1/20) * (u[0]**2 + u[1]**2)


def derivatives(u, g):
    f1 = -2 * u[0]**2 * exp(-u[1]**2 - u[0]**2) * log(exp(1)) + exp(-u[1]**2 - u[0]**2) + u[0]/10
    f2 = u[1]/10 - 2 * u[0] * u[1] * exp(-u[1]**2 - u[0]**2) * log(exp(1))
    return np.array([f1, f2])


def sec_derivatives(u, g):
    u1 = u[0]
    u2 = u[1]
    f1 = 4 * u[0]**3 * exp(-u[1]**2 - u[0]**2) * (log(exp(1)))**2 - 6 * u[0] * exp(-u[1]**2 - u[0]**2) * log(exp(1)) + 1/10
    f2 = 4 * u[0]**2 * u[1] * exp(-u[1]**2 - u[0]**2) * (log(exp(1)))**2 - 2 * u[1] * exp(-u[1]**2 - u[0]**2) * log(exp(1))
    f3 = 4 * u[0]**2 * u[1] * exp(-u[1]**2 - u[0]**2) * (log(exp(1)))**2 - 2 * u[1] * exp(-u[1]**2 - u[0]**2) * log(exp(1))
    f4 = 4 * u[0] * u[1]**2 * exp(-u[1]**2 - u[0]**2) * (log(exp(1)))**2 - 2 * u[0] * exp(-u[1]**2 - u[0]**2) * log(exp(1)) + 1/10
    return np.array([[f1, f2], [f3, f4]])
