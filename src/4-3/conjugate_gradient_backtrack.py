import numpy as np
from math import sqrt
import math
from src.models import model_4a
from src.models import model_one


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
            # beta = np.dot(f_new, f_new) / np.dot(f_old, f_old)
            # take 4.8
            a = np.dot(f_new.T, (f_new - f_old))
            b = np.dot(h_old.T, (f_new - f_old))
            #print(f"f_new = {f_new}")
            #print(f"f_old = {f_old}")
            #print(f"Value for a: {a}")
            #print(f"Value for b: {b}")
            beta = a/b

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


if __name__ == "__main__":
    a = np.array([1, 1])
    g = np.array([1, 1])
    # Fixed stepsize
    alpha = 0.04
    # Initial guess
    # u_initial = np.array([-0.7, 0.75])
    u_initial = np.array([0, 0])
    # Dunno, set as I liked for c and r: 0 < c < 1, 0 < r < 1
    c = 0.5
    r = 0.5
    n_reset = 25

    print(conjugate_gradient(a, g, alpha, u_initial, model_one.model, model_one.derivatives, c, r, n_reset))
    # print(conjugate_gradient(a, g, alpha, u_initial, rosenbrock_function, rosenbrock_derivatives, c, r, n_reset))
    u80 = np.zeros(80)
    g80 = np.zeros(80)
    # -1 because my matrix goes from 0 to 79, and not from 1 to 80
    g80[62 - 1] = 1
    g80[79 - 1] = 1
    print(conjugate_gradient(a, g80, alpha, u80, model_4a.model, model_4a.derivatives, c, r, n_reset))



