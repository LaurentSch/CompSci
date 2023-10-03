import numpy as np

# Define the Rosenbrock function m(u)
def rosenbrock(u):
    return (1 - u[0])**2 + 100 * (u[1] - u[0]**2)**2

# Initial parameters
a = 0.04  # Step size
u = np.array([-0.75, 0.7])  # Initial guess

# Steepest Descent Algorithm
mold = 10**100  # Dummy value
while True:
    mnew = rosenbrock(u)
    if mnew >= mold:
        break

    mold = mnew

    # Compute the gradient of the Rosenbrock function
    df_du = np.array([-2 * (1 - u[0]) - 400 * u[0] * (u[1] - u[0]**2),
                      200 * (u[1] - u[0]**2)])

    # Calculate the descent direction
    h = -df_du

    # Update u using the step size and descent direction
    u = u + a * h

# Completion
m_star = mold
u_star = u - a * h

print("Optimal m*:", m_star)
print("Optimal u*:", u_star)
