import numpy as np
from src.models import model_4a
from src.Newtons_method import newtons_method


def BFGS_strong(a, g, alpha_init, u_initial, tol, model, derivative, c_1, c_2, r):
    # Initialization
    n_var = len(u_initial)  # Set parameters of the objective function
    u = u_initial
    L_new = np.eye(n_var)

    # Quasi-Newton Algorithm
    m_new = model(u, g)
    f_2 = derivative(u, g)
    # initialize m_old and f_new with dummy values
    m_old = 10**100
    f_new = np.zeros(n_var)
    # counter
    cnt = 0
    # dummy delta_u
    delta_u = np.array([0, 0])
    while m_new < m_old:
        m_old = m_new
        f_old = f_new
        L_old = L_new
        f_new = f_2
        delta_f = f_new - f_old

        # Determine search direction
        if cnt == 0:
            h = -np.dot(L_old, f_new)
        else:
            delta_f = delta_f.reshape((len(delta_u), 1))
            part_1 = (np.dot(delta_u.T, delta_f) + np.dot(np.dot(delta_f.T, L_old), delta_f)) * np.outer(delta_u, delta_u.T) / np.dot(delta_u.T, delta_f)**2
            part_2 = (np.outer(L_old.dot(delta_f).flatten(), delta_u) + np.outer(delta_u.flatten(), delta_f.T.dot(L_old))) / np.dot(delta_u.T, delta_f)
            L_new = L_old + part_1 - part_2
            h = -np.dot(L_new, f_new)

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

        # Complete iteration
        delta_u = u_x - u
        u = u_x
        cnt += 1

        if signal_1 == 1:
            m_new = m_3
            f_2 = f_3
        else:
            m_new = m_2

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
    print(BFGS_strong(a, g80, alpha, u80, tolerance, model_4a.model, model_4a.derivatives, c1, c2, r))
