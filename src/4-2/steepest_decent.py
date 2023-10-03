import numpy as np
import math


def steepest_decent():
    a = [1, 1]
    g = [1, 1]
    # Initial stepsize
    alpha = 0.4
    # Dunno, set as I liked for c and r: 0 < c < 1, 0 < r < 1
    c = 0.3
    r = 0.3
    # Initial guess
    u = np.array([-0.75, 0.7])

    m_new = rosenbrock_function(u)
    m_old = 10**100
    # initialize h with dummy values
    h = np.array([0, 0])

    # To check
    count = 0
    while m_new < m_old:
        m_old = m_new
        # partial derivatives
        f1 = 20*(u[0]-u[1]**2)+2*(u[0]-1)
        f2 = -40*u[1]*(u[0]-u[1]**2)
        f = np.array([f1, f2])

        h = -f

        # backtracking line search
        alpha = 1/r * alpha
        m_x = 10**100   # dummy value
        # numpy transpose matrix notation = matrix.T
        while m_x > (m_new + c * alpha * f * h.T).all():
            alpha = r * alpha # decrease stepsize by factor r
            u_x = u + alpha * h
            m_x = rosenbrock_function(u_x)
        m_new = m_x
        u = u_x

    # COMPLETION
    m_star = m_old
    u_star = u - alpha * h
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
    m = a[0] * (math.sqrt( (1+u[0])**2 + (1+u[1])**2 ) - math.sqrt(2) )**2 + \
        a[1] * (math.sqrt( (1-u[0])**2 + (1+u[1])**2 ) - math.sqrt(2) )**2 - \
        g[0]*u[0] - g[1]*u[1]
    return m


def steepest_decent_mod_one():
    a = [1, 1]
    g = [1, 1]
    # Initial stepsize
    alpha = 0.4
    # Dunno, set as I liked for c and r: 0 < c < 1, 0 < r < 1
    c = 0.3
    r = 0.3
    # Initial guess
    u = np.array([-0.75, 0.7])

    m_new = model_one(u)
    m_old = 10**100
    # initialize h with dummy values
    h = np.array([0, 0])

    # To check
    count = 0
    while m_new < m_old:
        m_old = m_new
        # partial derivatives
        f1 = 2*(u[0] - 1)*(math.sqrt((1 - u[0])**2 + (u[1] + 1)**2) - math.sqrt(2))/math.sqrt((1 - u[0])**2 + (u[1] + 1)**2) + 2*(u[0] + 1)*(math.sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - math.sqrt(2))/math.sqrt((u[0] + 1)**2 + (u[1] + 1)**2)
        f2 = 2*(u[1] + 1)*(math.sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - math.sqrt(2))/math.sqrt((u[0] + 1)**2 + (u[1] + 1)**2) + 2*(u[1] + 1)*(math.sqrt((1 - u[0])**2 + (u[1] + 1)**2) - math.sqrt(2))/math.sqrt((1 - u[0])**2 + (u[1] + 1)**2)

        f = np.array([f1, f2])

        h = -f

        # backtracking line search
        alpha = 1/r * alpha
        m_x = 10**100   # dummy value
        # numpy transpose matrix notation = matrix.T
        while m_x > (m_new + c * alpha * f * h.T).all():
            alpha = r * alpha # decrease stepsize by factor r
            u_x = u + alpha * h
            m_x = model_one(u_x)
        m_new = m_x
        u = u_x

    # COMPLETION
    m_star = m_old
    u_star = u - alpha * h
    return m_star, u_star


print(steepest_decent())
print(steepest_decent_mod_one())
