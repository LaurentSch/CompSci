import numpy as np
from numpy.linalg import solve, norm
from src.models import model_71
from math import sqrt


# Augmented objective function with penalty term
def augmented_objective(u, a, g, rho, model, R, x_R, y_R):
    return model(u, g) + 0.5 * rho * np.sum(constraint_function(u, R, x_R, y_R)**2)


# Define your constraint function h(u)
def constraint_function(u, R, x_R, y_R):
    j_values = np.arange(31, 41)
    distances = np.sqrt((x_R - j_values + 31 - u[0::2])**2 + (y_R - 4 - u[1::2])**2) - R
    return distances


def newton_with_penalty(a, g, u, tol, model, derivative, sec_deriv, h, rho, max_iterations):
    cnt = 0
    while True:
        if cnt >= max_iterations:
            print("Max iteration reached. Exiting")
            break

        f = derivative(u, g)
        k = sec_deriv(u, g)
        hessian_penalty = rho * sec_deriv(h(u), g)
        hessian_combined = k + hessian_penalty
        h = np.linalg.solve(hessian_combined, -f)
        u = u + h

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
u_initial = np.array([0.5, 0.2])
tolerance = 1e-12
max_iterations = 100
rho = 1.0  # Adjust the penalty parameter



# Usage of the newton_with_penalty function
result = newton_with_penalty(a, g, u_initial, tolerance, model_71.model, model_71.derivatives, model_71.sec_derivatives, constraint_function, rho, max_iterations)
print(result)
