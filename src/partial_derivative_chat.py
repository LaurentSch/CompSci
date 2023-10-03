import numpy as np
import math
import sympy as sp

# Symbols are slow af, so never use them in the main program

# Define symbolic variables
u0, u1 = sp.symbols('u0 u1')

# Define the function m
g = np.array([0, 0])
a = np.array([1, 1])
m = a[0] * (sp.sqrt((1 + u0)**2 + (1 + u1)**2) - sp.sqrt(2))**2 + \
    a[1] * (sp.sqrt((1 - u0)**2 + (1 + u1)**2) - sp.sqrt(2))**2 - \
    g[0] * u0 - g[1] * u1

# Calculate the partial derivatives
partial_m_u0 = sp.diff(m, u0)  # ∂m/∂u[0]
partial_m_u1 = sp.diff(m, u1)  # ∂m/∂u[1]

# Print the results
print("Partial derivative ∂m/∂u[0]:", partial_m_u0)
print("Partial derivative ∂m/∂u[1]:", partial_m_u1)
