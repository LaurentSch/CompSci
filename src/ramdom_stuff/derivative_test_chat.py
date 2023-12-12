import math

import sympy as sp

# Define the variables and the function
u0, u1 = sp.symbols('u0 u1')
g = [1, 1]
a = [1, 1]
f = (sp.sqrt((1 + u0)**2 + (1 + u1)**2) - sp.sqrt(2))**2 + \
    (sp.sqrt((1 - u0)**2 + (1 + u1)**2) - sp.sqrt(2))**2 - \
    g[0] * u0 - g[1] * u1


# Calculate the partial derivative of f with respect to x
df_dx = sp.diff(f, u0)

# Calculate the partial derivative of f with respect to y
df_dy = sp.diff(f, u1)

# Display the results
print("Partial derivative of f with respect to x:")
print(df_dx)

print("Partial derivative of f with respect to y:")
print(df_dy)
