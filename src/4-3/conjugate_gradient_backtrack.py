import numpy as np
from math import sqrt
import math


def conjugate_gradient(a, g, alpha_init, u_initial, model, derivatives, c, r, n_reset):

    # Initial guess
    u = u_initial

    m_new = model(u, g)
    alpha = alpha_init
    # dummy values and counter
    m_old = 10**100
    f_new = np.array([0, 0])
    h_new = np.array([0, 0])
    # Counter
    cnt = 0

    while m_new < m_old:
        m_old = m_new
        f_old = f_new
        h_old = h_new
        # partial derivatives, gradient
        f_new = derivatives(u, g)
        if cnt % n_reset == 0:
            h_new = -f_new
        else:
            # beta = f_new.T * f_new / f_old.T * f_old
            # take 4.6
            beta = np.dot(f_new, f_new) / np.dot(f_old, f_old)
            print(beta)
            # confused about pseudo-code. Is it max(0, beta), or np.maximum([0, beta] * h_old)
            h_new = -f_new + np.maximum(0, beta) * h_old

        # backtracking line search
        alpha = 1/r * alpha_init
        m_x = 10**100   # dummy value
        # numpy transpose matrix notation = matrix.T
        while m_x > m_new + c * alpha * np.dot(h_new, f_new):
            alpha = r * alpha # decrease stepsize by factor r
            u_x = u + alpha * h_new
            m_x = model(u_x, g)
        m_new = m_x
        u = u_x
        cnt += 1
    # COMPLETION
    m_star = m_old
    u_star = u - alpha * h_new
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
    g = np.array([0, 0])
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
u_initial = np.array([-0.7, 0.75])
# Dunno, set as I liked for c and r: 0 < c < 1, 0 < r < 1
c = 0.4
r = 0.5
n_reset = 25

print(conjugate_gradient(a, g, alpha, u_initial, model_one, model_one_derivatives, c, r, n_reset))
#print(conjugate_gradient(a, g, alpha, u_initial, rosenbrock_function, rosenbrock_derivatives, c, r, n_reset))
