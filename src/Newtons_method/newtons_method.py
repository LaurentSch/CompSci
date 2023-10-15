import numpy as np
from numpy import sqrt
from src.models import model_4a
from src.models import model_one


def newton(a, g, u, tol, model, derivative, sec_deriv):
    f = derivative(u, g)
    while sqrt(np.dot(f.T, f)) > tol:
        print(sqrt(np.dot(f.T, f)))
        k = sec_deriv(u, g)
        h = np.linalg.solve(k, -f)
        u = u + h
        f = derivative(u, g)
    u_star = u
    m_star = model(u_star, g)
    return m_star, u_star


if __name__ == "__main__":
    a = np.array([1, 1])
    g = np.array([1, 1])
    u = np.array([0, 0])
    tol = 10**(-12)
    # print(newton(a, g, u, tol, model_one.model, model_one.derivatives, model_one.sec_derivatives))
    a80 = np.ones(80)
    u80 = np.zeros(80)
    g80 = np.zeros(80)
    # -1 because my matrix goes from 0 to 79, and not from 1 to 80
    g80[62 - 1] = 1
    g80[79 - 1] = 1
    print(newton(a, g80, u80, tol, model_4a.model, model_4a.derivatives, model_4a.sec_derivatives))


