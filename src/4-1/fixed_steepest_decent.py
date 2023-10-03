import numpy as np
import math


def fixed_steepest_decent():
    # INITIALISATION
    # Set a, g % set parameters of the objective function
    a = np.array([1, 1])
    g = np.array([1, 1])
    # Fixed stepsize
    alpha = 0.04
    # Initial guess
    u = np.array([-0.75, 0.7])

    # STEEPEST DESCENT ALGORITHM
    # Compute mnew = m(u)
    m_new = rosenbrock_function(u)
    # mold = 10100 % dummy value
    m_old = 10**100
    # initialize h with dummy values
    h = np.array([0, 0])

    # while mnew < mold
    # mold = mnew
    # Compute f = ∂m
    # h = −f
    # u = u + α h
    # Compute mnew = m(u)
    count = 0
    while m_new < m_old:
        m_old = m_new
        # partial derivatives
        f1 = 20*(u[0]-u[1]**2)+2*(u[0]-1)
        f2 = -40*u[1]*(u[0]-u[1]**2)
        f = np.array([f1, f2])

        h = -f

        # update u
        u = u + alpha * h
        m_new = rosenbrock_function(u)
        count += 1
        print(count)

    # COMPLETION
    # m∗ = mold, u∗ = u − α h
    m_star = m_old
    u_star = u
    return m_star, u_star


def rosenbrock_function(u):
    # m(u) = 10(u2 − (u1)^2)^2 + (u1 − 1)^2.
    m = 10 * (u[0] - u[1]**2)**2 + (u[0] - 1)**2
    return m


def model_one(u):
    # generally g = 0, 0
    # a = 1,1
    g = np.array([0, 0])
    a = np.array([1, 1])
    m = a[0] * (math.sqrt((1+u[0])**2 + (1+u[1])**2) - math.sqrt(2) )**2 + \
        a[1] * (math.sqrt((1-u[0])**2 + (1+u[1])**2) - math.sqrt(2) )**2 - \
        g[0]*u[0] - g[1]*u[1]
    return m


def fixed_steepest_decent_mod_one():
    # INITIALISATION
    # Set a, g % set parameters of the objective function
    a = np.array([1, 1])
    g = np.array([1, 1])
    # Fixed stepsize
    alpha = 0.004
    # Initial guess
    u = np.array([0, 0])

    # STEEPEST DESCENT ALGORITHM
    # Compute mnew = m(u)
    m_new = model_one(u)
    # mold = 10100 % dummy value
    m_old = 10**100
    # initialize h with dummy values
    h = np.array([0, 0])

    # while mnew < mold
    # mold = mnew
    # Compute f = ∂m
    # h = −f
    # u = u + α h
    # Compute mnew = m(u)
    count = 0
    while m_new < m_old:
        m_old = m_new
        # partial derivatives
        f1 = 2*(u[0] - 1)*(math.sqrt((1 - u[0])**2 + (u[1] + 1)**2) - math.sqrt(2))/math.sqrt((1 - u[0])**2 + (u[1] + 1)**2) + 2*(u[0] + 1)*(math.sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - math.sqrt(2))/math.sqrt((u[0] + 1)**2 + (u[1] + 1)**2)
        f2 = 2*(u[1] + 1)*(math.sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - math.sqrt(2))/math.sqrt((u[0] + 1)**2 + (u[1] + 1)**2) + 2*(u[1] + 1)*(math.sqrt((1 - u[0])**2 + (u[1] + 1)**2) - math.sqrt(2))/math.sqrt((1 - u[0])**2 + (u[1] + 1)**2)

        f = np.array([f1, f2])

        h = -f

        # update u
        u = u + alpha * h
        m_new = model_one(u)
        count += 1
        print(count)

    # COMPLETION
    # m∗ = mold, u∗ = u − α h
    m_star = m_old
    u_star = u
    return m_star, u_star


print(fixed_steepest_decent())
print(fixed_steepest_decent_mod_one())
