import numpy as np
from numpy.linalg import solve, norm
from src.models import model_71
from math import sqrt


def newton(a, g, u, tol, model, derivative, sec_deriv, max_iterations):
    f = derivative(u, g)
    cnt = 0
    while sqrt(np.dot(f.T, f)) > tol:
        if cnt >= max_iterations:
            print("Max iteration reached. Exiting")
            break
        k = sec_deriv(u, g)
        h = np.linalg.solve(k, -f)
        u = u + h
        f = derivative(u, g)
        if norm(f) < tol:
            print("Stationary point found: Local minimum")
            break
        cnt += 1
        print(cnt)
    u_star = u
    m_star = model(u_star, g)
    return m_star, u_star


# Example usage with a random initial guess
a = None
g = None
# Use a random initial guess in domain (−1, 1) × (−1, 1)
u_initial = np.array([-0.5, 0.2])
tolerance = 10**(-12)
max_iterations = 100  # Adjust the maximum number of iterations

print(newton(a, g, u_initial, tolerance, model_71.model, model_71.derivatives, model_71.sec_derivatives, max_iterations))
