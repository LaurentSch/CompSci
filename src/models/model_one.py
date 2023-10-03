import math


def model(u, a, g):
    term1 = a[0] * (math.sqrt((1 + u[0])**2 + (1 + u[1])**2) - math.sqrt(2))**2
    term2 = a[1] * (math.sqrt((1 - u[0])**2 + (1 + u[1])**2) - math.sqrt(2))**2
    term3 = g[0] * u[0]
    term4 = g[1] * u[1]

    m = term1 + term2 - term3 - term4
    return m

# Example usage:
u = [0, 0]  # Replace with your desired values of u
a = [1, 1]  # Replace with your values of a
g = [0, 0]  # Replace with your values of g

result = model(u, a, g)
print("Result of the model:", result)
