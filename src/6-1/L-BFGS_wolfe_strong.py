import numpy as np
from src.models import model_4a, model_one
from src.Newtons_method import newtons_method


def LBFGS_standard(a, g, alpha_init, u_initial, tol, model, derivative, c_1, c_2, r):
    # Initialization
    n_var = len(u_initial)  # Set parameters of the objective function
    u = u_initial
    L_0 = np.eye(n_var)
    n_li = 5 # standard values are between 5 and 20
    delta_F = np.zeros((n_var, n_li))
    delta_U = np.zeros((n_var, n_li))
    rho = np.zeros(n_li)

    # Quasi-newton algorithm
    m_new = model(u, g)
    f_2 = derivative(u, g)
    # initialize m_old and f_new with dummy values
    m_old = 10**100
    # counter
    cnt = 0
    while m_new < m_old:
        m_old = m_new
        f_new = f_2
        cnt += 1
        # determine search direction
        if cnt == 1:
            h = - np.dot(L_0, f_new)
        else:
            gamma = np.zeros(n_li)
            h = f_new
            upper_bound = max(1, n_li - cnt + 2)
            # not 100% sure if it's (- 1) or not
            for j in range(n_li, upper_bound - 1, -1):
                # not sure if j-1 is correct
                p = rho[j-1]
                delta_u = delta_U[:, j-1]
                delta_f = delta_F[:, j-1]
                y = p * np.dot(delta_u.T, h)
                h = h - y * delta_f
                gamma[j-1] = y
            h = np.dot(L_0, h)
            for j in range(max(0, n_li - cnt + 2), n_li):
                p = rho[j-1]
                delta_u = delta_U[:, j-1]
                delta_f = delta_F[:, j-1]
                y = gamma[j-1]
                # wanted to call it n, but maybe confusing with n_li existing
                eta = p * np.dot(delta_f.T, h)
                h = h + (y - eta) * delta_u
            h = -h

        # --------------------------------
        # Line search
        # determine initial search domain, but stop if acceptable stepsize is found
        signal_1 = 0
        alpha_3 = alpha_init
        u_x = u + alpha_3 * h
        m_3 = model(u_x, g)
        f_3 = derivative(u_x, g)

        if (m_3 <= (m_new + c_1 * alpha_3 * np.dot(h.T, f_new))) and np.abs(np.dot(h.T, f_3)) <= c_2 * np.abs(np.dot(h.T, f_new)):
            signal_1 = 1

        while (m_3 < (m_new + c_1 * alpha_3 * np.dot(h.T, f_new))) and np.dot(h.T, f_3) < - c_2 * np.dot(h.T, f_new) and (signal_1 == 0):
            alpha_3 = alpha_3 / r
            u_x = u + alpha_3 * h
            m_3 = model(u_x, g)
            f_3 = derivative(u_x, g)
            if (m_3 <= (m_new + c_1 * alpha_3 * np.dot(h.T, f_new))) and np.abs(np.dot(h.T, f_3)) <= (c_2 * np.abs(np.dot(h.T, f_new))):
                signal_1 = 1

        # Apply bisection method if no acceptable stepsize is found yet
        if signal_1 == 0:
            signal_2 = 0
            alpha_1 = 0
            m_1 = m_new
            f_1 = f_new
            alpha_2 = alpha_3 / 2
            u_x = u + alpha_2 * h
            m_2 = model(u_x, g)
            f_2 = derivative(u_x, g)

            while signal_2 == 0:
                if alpha_3 - alpha_1 < tol:
                    signal_2 = 1
                    m_2 = m_new
                    f_2 = f_new
                elif m_2 > (m_new + c_1 * alpha_2 * np.dot(h.T, f_new)):
                    alpha_3 = alpha_2
                    m_3 = m_2
                    f_3 = f_2
                    alpha_2 = (alpha_1 + alpha_2) / 2
                    u_x = u + alpha_2 * h
                    m_2 = model(u_x, g)
                    f_2 = derivative(u_x, g)
                else:
                    if np.dot(h.T, f_2) < c_2 * np.dot(h.T, f_new):
                        alpha_1 = alpha_2
                        m_1 = m_2
                        f_1 = f_2
                        alpha_2 = (alpha_2 + alpha_3) / 2
                        u_x = u + alpha_2 * h
                        m_2 = model(u_x, g)
                        f_2 = derivative(u_x, g)
                    elif np.dot(h.T, f_2) > - c_2 * np.dot(h.T, f_new):
                        alpha_3 = alpha_2
                        m_3 = m_2
                        f_3 = f_2
                        alpha_2 = (alpha_1 + alpha_2) / 2
                        m_2 = model(u_x, g)
                        f_2 = derivative(u_x, g)
                    else:
                        signal_2 = 1

        # --------------------------------
        # Complete iteration
        delta_u = u_x - u
        u = u_x
        if signal_1 == 1:
            m_new = m_3
            f_2 = f_3
        else:
            m_new = m_2

        delta_f = f_2 - f_new
        delta_U[:, 1: n_li-2] = delta_U[:, 2: n_li-1]
        delta_U[:, n_li-2] = delta_u

        delta_F[:, 1: n_li-2] = delta_F[:, 2: n_li-1]
        delta_F[:, n_li-2] = delta_f
        rho[1: (n_li-2)] = rho[2: n_li-1]
        rho[n_li-2] = 1 / np.dot(delta_f.T, delta_u)

    # Completion
    m_star = m_old
    u_star = u - delta_u
    return m_star, u_star


if __name__ == "__main__":
    alpha = 0.04
    c1 = 0.4
    c2 = 0.6
    r = 0.5
    a = np.ones(80)
    u80 = np.zeros(80)
    g80 = np.zeros(80)
    # -1 because my matrix goes from 0 to 79, and not from 1 to 80
    g80[62 - 1] = 1
    g80[79 - 1] = 1
    tolerance = 10**(-12)
    print(LBFGS_standard(a, g80, alpha, u80, tolerance, model_4a.model, model_4a.derivatives, c1, c2, r))
    # Newtons to compare
    print(newtons_method.newton(a, g80, u80, tolerance, model_4a.model, model_4a.derivatives, model_4a.sec_derivatives))

    # a = np.array([1, 1])
    # g = np.array([1, 1])
    # # Initial guess
    # u_initial = np.array([0, 0])
    # print(LBFGS_standard(a, g, alpha, u_initial, tolerance, model_one.model, model_one.derivatives, c1, c2, r))
    # # Newtons to compare
    # print(newtons_method.newton(a, g, u_initial, tolerance, model_one.model, model_one.derivatives, model_one.sec_derivatives))

