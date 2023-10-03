def fixed_steepest_decent():
    # INITIALISATION
    # Set parameters of the objective function
    a = 1
    g = 1

    # Set step size and initial guess
    alpha = 0.04
    u = [-0.75, 0.7]

    # Compute mnew = m(u)
    m_new = rosenbrock_function(u)

    # Initialize mold with a dummy value
    m_old = float('inf')

    # STEEPEST DESCENT ALGORITHM
    while m_new < m_old:
        # Update mold to mnew
        m_old = m_new

        # Compute partial derivatives
        f1 = 20 * (u[0] - u[1]**2) + 2 * (u[0] - 1)
        f2 = -40 * u[1] * (u[0] - u[1]**2)

        # Compute the search direction
        h = [-f1, -f2]

        # Update u
        u[0] = u[0] + alpha * h[0]
        u[1] = u[1] + alpha * h[1]

        # Compute mnew = m(u)
        m_new = rosenbrock_function(u)

    # COMPLETION
    # m∗ = mold, u∗ = u - αh
    m_star = m_old
    u_star = [u[0] - alpha * h[0], u[1] - alpha * h[1]]

    return m_star, u_star


def rosenbrock_function(u):
    # m(u) = 10(u2 − (u1)^2)^2 + (u1 − 1)^2.
    m = 10 * (u[0] - u[1]**2)**2 + (u[0] - 1)**2
    return m

a = fixed_steepest_decent()
# Print results
print("Minimum value of the Rosenbrock function:", a[0])
print("Optimal u:", a[1])
