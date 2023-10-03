import numpy as np
from math import sqrt
import math

def fixed_steepest_decent(a, g, alpha, u_initial, model, derivatives):
    u = u_initial
    m_new = model(u, g)
    # mold = 10100 % dummy value
    m_old = 10**100
    count = 0
    while m_new < m_old:
        m_old = m_new
        # partial derivatives, gradient
        f = derivatives(u, g)

        h = -f

        # update u
        u = u + alpha * h
        m_new = model(u, g)
        count += 1
        print(count)
    m_star = m_old
    u_star = u - alpha * h
    return m_star, u_star


def rosenbrock_function(u, g):
    # m(u) = 10(u2 − (u1)**2)**2 + (u1 − 1)**2.
    m = 10 * (u[0] - u[1]**2)**2 + (u[0] - 1)**2
    return m


def rosenbrock_derivatives(u, g):
    f1 = 20*(u[0]-u[1]**2)+2*(u[0]-1)
    f2 = -40*u[1]*(u[0]-u[1]**2)
    return np.array([f1, f2])


# change g to arguments
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
u = np.array([-0.7, 0.75])

f1 = 20*(u[0]-u[1]**2)+2*(u[0]-1)
f2 = -40*u[1]*(u[0]-u[1]**2)
f = np.array([f1, f2])
print(fixed_steepest_decent(a, g, alpha, u, model_one, model_one_derivatives))
# print(fixed_steepest_decent(a, g, alpha, u, rosenbrock_function, rosenbrock_derivatives))
