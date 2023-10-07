import numpy as np
from math import sqrt
from src.models import model_4a
from src.models import model_one


def newton(a, g, u, tol, model, derivative, sec_deriv):
    f = derivative(u, g)
    while sqrt(np.dot(f.T, f)) > tol:
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
    print(newton(a, g, u, tol, model_one.model, model_one.derivatives, model_one.sec_derivatives))
