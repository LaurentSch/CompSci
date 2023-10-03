import numpy as np
from math import sqrt
import math


def steepest_decent(a, g, alpha_init, u_initial, model, derivatives, c, r):

    # Initial guess
    u = u_initial

    m_new = model(u, g)
    m_old = 10**100
    # initialize h with dummy values
    h = np.array([0, 0])

    # To check
    count = 0
    while m_new < m_old:
        m_old = m_new
        # partial derivatives, gradient
        f = derivatives(u, g)

        h = -f

        # backtracking line search
        alpha = 1/r * alpha_init
        m_x = 10**100   # dummy value
        # numpy transpose matrix notation = matrix.T
        while m_x > m_new + c * alpha * np.dot(h, f):
            alpha = r * alpha # decrease stepsize by factor r
            u_x = u + alpha * h
            m_x = model(u_x, g)
        m_new = m_x
        u = u_x
    # COMPLETION
    m_star = m_old
    u_star = u - alpha * h
    return m_star, u_star


def rosenbrock_function(u, g):
    # m(u) = 10(u2 − (u1)^2)^2 + (u1 − 1)^2.
    m = 10 * (u[0] - u[1]**2)**2 + (u[0] - 1)**2
    return m


def rosenbrock_derivatives(u, g):
    f1 = 20*(u[0]-u[1]**2)+2*(u[0]-1)
    f2 = -40*u[1]*(u[0]-u[1]**2)
    return np.array([f1, f2])


def model_one(u, g):
    # generally g = 0, 0
    # a = 1,1
    a = np.array([1, 1])
    m = a[0] * (math.sqrt((1+u[0])**2 + (1+u[1])**2) - math.sqrt(2) )**2 + \
        a[1] * (math.sqrt((1-u[0])**2 + (1+u[1])**2) - math.sqrt(2) )**2 - \
        g[0]*u[0] - g[1]*u[1]
    return m


def model_one_derivatives(u, g):
    u1 = u[0]
    u2 = u[1]
    g1 = g[0]
    g2 = g[1]
    a1 = 1
    a2 = 1
    f1 = (2*a1*(u1+1)*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/sqrt((u2+1)**2+(u1+1)**2) - (2*a2*(1-u1)*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/sqrt((u2+1)**2+(1-u1)**2)-g1
    f2 = (2*a1*(u2+1)*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/sqrt((u2+1)**2+(u1+1)**2) + (2*a2*(u2+1)*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/sqrt((u2+1)**2+(1-u1)**2)-g2
    return np.array([f1, f2])


a = np.array([1, 1])
g = np.array([1, 1])
# Fixed stepsize
alpha = 0.004
# Initial guess
u = np.array([-0.7, 0.7])
# Dunno, set as I liked for c and r: 0 < c < 1, 0 < r < 1
c = 0.3
r = 0.3

f1 = 20*(u[0]-u[1]**2)+2*(u[0]-1)
f2 = -40*u[1]*(u[0]-u[1]**2)
f = np.array([f1, f2])
print(steepest_decent(a, g, alpha, u, model_one, model_one_derivatives, c, r))
print(steepest_decent(a, g, alpha, u, rosenbrock_function, rosenbrock_derivatives, c, r))
