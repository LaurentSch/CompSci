import numpy as np
from math import sqrt, floor
import matplotlib.pyplot as plt


def model(u, g):
    m = -np.dot(g, u)

    for i in range(10):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j - 1) - 1]
        u2 = u[(2 * j) - 1]
        m += (sqrt(u1 ** 2 + (1 + u2) ** 2) - 1) ** 2

    for i in range(9):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 1) - 1]
        u2 = u[(2 * j + 2) - 1]
        m += (sqrt((1 + u1) ** 2 + (1 + u2) ** 2) - sqrt(2)) ** 2

    for i in range(9):
        # part 3 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j - 1) - 1]
        u2 = u[(2 * j) - 1]
        m += (sqrt((1 - u1) ** 2 + (1 + u2) ** 2) - sqrt(2)) ** 2

    for i in range(30):
        # part 4 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 20) - 1]
        u2 = u[(2 * j) - 1]
        u3 = u[(2 * j + 19) - 1]
        u4 = u[(2 * j - 1) - 1]
        m += (sqrt((1 + u1 - u2) ** 2 + (u3 - u4) ** 2) - 1) ** 2

    for i in range(36):
        # part 5 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i
        u1 = u[(2 * j + 1 + 2 * floor((j - 1) / 9)) - 1]
        u2 = u[(2 * j - 1 + 2 * floor((j - 1) / 9)) - 1]
        u3 = u[(2 * j + 2 + 2 * floor((j - 1) / 9)) - 1]
        u4 = u[(2 * j + 2 * floor((j - 1) / 9)) - 1]
        m += (sqrt((1 + u1 - u2) ** 2 + (u3 - u4) ** 2) - 1) ** 2

    for i in range(27):
        # part 6 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 21 + 2 * floor((j - 1) / 9)) - 1]
        u2 = u[(2 * j - 1 + 2 * floor((j - 1) / 9)) - 1]
        u3 = u[(2 * j + 22 + 2 * floor((j - 1) / 9)) - 1]
        u4 = u[(2 * j + 2 * floor((j - 1) / 9)) - 1]
        m += (sqrt((1 + u1 - u2) ** 2 + (1 + u3 - u4) ** 2) - sqrt(2)) ** 2

    for i in range(27):
        # part 7 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 19 + 2 * floor((j - 1) / 9)) - 1]
        u2 = u[(2 * j + 1 + 2 * floor((j - 1) / 9)) - 1]
        u3 = u[(2 * j + 20 + 2 * floor((j - 1) / 9)) - 1]
        u4 = u[(2 * j + 2 + 2 * floor((j - 1) / 9)) - 1]
        m += (sqrt((1 - u1 + u2) ** 2 + (1 + u3 - u4) ** 2) - sqrt(2)) ** 2

    return m


# g = np.zeros(80)
def derivatives(u, g):
    f_new = - g
    for i in range(10):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j - 1) - 1]
        u2 = u[(2 * j) - 1]
        # derivative for u[2*i-1]
        f1 = (2 * u1 * (sqrt((u2 + 1) ** 2 + u1 ** 2) - 1)) / sqrt((u2 + 1) ** 2 + u1 ** 2)
        # derivative for u[2*i]
        f2 = (2 * (u2 + 1) * (sqrt((u2 + 1) ** 2 + u1 ** 2) - 1)) / sqrt((u2 + 1) ** 2 + u1 ** 2)
        # Add results to the gradient
        f_new[(2 * j - 1) - 1] += f1
        f_new[(2 * j) - 1] += f2

    for i in range(9):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 1) - 1]
        u2 = u[(2 * j + 2) - 1]
        # derivative for u[2*i-1]
        f1 = (2 * (u1 + 1) * (sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) - sqrt(2))) / sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2)
        # derivative for u[2*i]
        f2 = (2 * (u2 + 1) * (sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) - sqrt(2))) / sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2)
        # Add results to the gradient
        f_new[(2 * j + 1) - 1] += f1
        f_new[(2 * j + 2) - 1] += f2

    for i in range(9):
        # part 3 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j - 1) - 1]
        u2 = u[(2 * j) - 1]
        # derivative for u[2*i-1]
        f1 = -((2 * (1 - u1) * (sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) - sqrt(2))) / sqrt((u2 + 1) ** 2 + (1 - u1) ** 2))
        # derivative for u[2*i]
        f2 = (2 * (u2 + 1) * (sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) - sqrt(2))) / sqrt((u2 + 1) ** 2 + (1 - u1) ** 2)
        # Add results to the gradient
        f_new[(2 * j - 1) - 1] += f1
        f_new[(2 * j) - 1] += f2

    for i in range(30):
        # part 4 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 20) - 1]
        u2 = u[(2 * j) - 1]
        u3 = u[(2 * j + 19) - 1]
        u4 = u[(2 * j - 1) - 1]
        # derivative for u[2*i-1]
        f1 = (2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt(
            (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        # derivative for u[2*i]
        f2 = -((2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt(
            (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2))
        f3 = (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / sqrt(
            (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        f4 = -((2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / sqrt(
            (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2))
        # Add results to the gradient
        f_new[(2 * j + 20) - 1] += f1
        f_new[(2 * j) - 1] += f2
        f_new[(2 * j + 19) - 1] += f3
        f_new[(2 * j - 1) - 1] += f4

    for i in range(36):
        # part 5 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 1 + 2 * floor((j - 1) / 9)) - 1]
        u2 = u[(2 * j - 1 + 2 * floor((j - 1) / 9)) - 1]
        u3 = u[(2 * j + 2 + 2 * floor((j - 1) / 9)) - 1]
        u4 = u[(2 * j + 2 * floor((j - 1) / 9)) - 1]
        # derivative for u[2*i-1]
        f1 = (2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        # derivative for u[2*i]
        f2 = -((2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        f3 = (2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        f4 = -((2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        # Add results to the gradient
        f_new[(2 * j + 1 + 2 * floor((j - 1) / 9)) - 1] += f1
        f_new[(2 * j - 1 + 2 * floor((j - 1) / 9)) - 1] += f2
        f_new[(2 * j + 2 + 2 * floor((j - 1) / 9)) - 1] += f3
        f_new[(2 * j + 2 * floor((j - 1) / 9)) - 1] += f4

    for i in range(27):
        # part 6 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 21 + 2 * floor((j - 1) / 9)) - 1]
        u2 = u[(2 * j - 1 + 2 * floor((j - 1) / 9)) - 1]
        u3 = u[(2 * j + 22 + 2 * floor((j - 1) / 9)) - 1]
        u4 = u[(2 * j + 2 * floor((j - 1) / 9)) - 1]
        # derivative for u[2*i-1]
        f1 = (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        # derivative for u[2*i]
        f2 = -((2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2))
        f3 = (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        f4 = -((2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2))
        # Add results to the gradient
        f_new[(2 * j + 21 + 2 * floor((j - 1) / 9)) - 1] += f1
        f_new[(2 * j - 1 + 2 * floor((j - 1) / 9)) - 1] += f2
        f_new[(2 * j + 22 + 2 * floor((j - 1) / 9)) - 1] += f3
        f_new[(2 * j + 2 * floor((j - 1) / 9)) - 1] += f4

    for i in range(27):
        # part 7 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 19 + 2 * floor((j - 1) / 9)) - 1]
        u2 = u[(2 * j + 1 + 2 * floor((j - 1) / 9)) - 1]
        u3 = u[(2 * j + 20 + 2 * floor((j - 1) / 9)) - 1]
        u4 = u[(2 * j + 2 + 2 * floor((j - 1) / 9)) - 1]
        # derivative for u[2*i-1]
        f1 = -((2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2))
        # derivative for u[2*i]
        f2 = (2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        f3 = (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        f4 = -((2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2))
        # Add results to the gradient
        f_new[(2 * j + 19 + 2 * floor((j - 1) / 9)) - 1] += f1
        f_new[(2 * j + 1 + 2 * floor((j - 1) / 9)) - 1] += f2
        f_new[(2 * j + 20 + 2 * floor((j - 1) / 9)) - 1] += f3
        f_new[(2 * j + 2 + 2 * floor((j - 1) / 9)) - 1] += f4
    return f_new


def sec_derivatives(u, g):
    # Initialize python matrix
    matrix_size = 80
    matrix = np.zeros(matrix_size ** 2).reshape((matrix_size, matrix_size))

    for i in range(10):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1i = (2 * j - 1) - 1
        u2i = (2 * j) - 1
        u1 = u[u1i]
        u2 = u[u2i]
        # "p1: Sec-Derivative u1, u1:"
        f1 = (2*(sqrt((u2+1)**2+u1**2)-1))/sqrt((u2+1)**2+u1**2)-(2*u1**2*(sqrt((u2+1)**2+u1**2)-1))/((u2+1)**2+u1**2)**(3/2)+(2*u1**2)/((u2+1)**2+u1**2)
        update_matrix_value(matrix, u1i, u1i, f1)

        # p1: Sec-Derivative u1, u2:
        f2 = (2*u1*(u2+1))/((u2+1)**2+u1**2)-(2*u1*(u2+1)*(sqrt((u2+1)**2+u1**2)-1))/((u2+1)**2+u1**2)**(3/2)
        update_matrix_value(matrix, u1i, u2i, f2)
        update_matrix_value(matrix, u2i, u1i, f2)

        # p1: Sec-Derivative u2, u1: Doublicate
        # f3 = (2 * u1 * (u2 + 1)) / ((u2 + 1) ** 2 + u1 ** 2) - (
        #         2 * u1 * (u2 + 1) * (sqrt((u2 + 1) ** 2 + u1 ** 2) - 1)) / ((u2 + 1) ** 2 + u1 ** 2) ** (3 / 2)

        # p1: Sec-Derivative u2, u2:
        f4 = (2*(sqrt((u2+1)**2+u1**2)-1))/sqrt((u2+1)**2+u1**2)-(2*(u2+1)**2*(sqrt((u2+1)**2+u1**2)-1))/((u2+1)**2+u1**2)**(3/2)+(2*(u2+1)**2)/((u2+1)**2+u1**2)
        update_matrix_value(matrix, u2i, u2i, f4)

    for i in range(9):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1i = (2 * j + 1) - 1
        u2i = (2 * j + 2) - 1
        u1 = u[u1i]
        u2 = u[u2i]

        # p2: Sec-Derivative u1, u1:
        f1 = (2*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/sqrt((u2+1)**2+(u1+1)**2)-(2*(u1+1)**2*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/((u2+1)**2+(u1+1)**2)**(3/2)+(2*(u1+1)**2)/((u2+1)**2+(u1+1)**2)
        update_matrix_value(matrix, u1i, u1i, f1)

        # p2: Sec-Derivative u1, u2:
        f2 = (2*(u1+1)*(u2+1))/((u2+1)**2+(u1+1)**2)-(2*(u1+1)*(u2+1)*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/((u2+1)**2+(u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u2i, f2)
        update_matrix_value(matrix, u2i, u1i, f2)

        # p2: Sec-Derivative u2, u2:
        f4 = (2*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/sqrt((u2+1)**2+(u1+1)**2)-(2*(u2+1)**2*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/((u2+1)**2+(u1+1)**2)**(3/2)+(2*(u2+1)**2)/((u2+1)**2+(u1+1)**2)
        update_matrix_value(matrix, u2i, u2i, f4)

    for i in range(9):
        # part 3 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1i = 2 * j - 1 - 1
        u2i = 2 * j - 1
        u1 = u[u1i]
        u2 = u[u2i]

        # p3: Sec-Derivative u1, u1:
        f1 = (2*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/sqrt((u2+1)**2+(1-u1)**2)-(2*(1-u1)**2*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/((u2+1)**2+(1-u1)**2)**(3/2)+(2*(1-u1)**2)/((u2+1)**2+(1-u1)**2)
        update_matrix_value(matrix, u1i, u1i, f1)
        # matrix[u1i][u1i] += f1

        # p3: Sec-Derivative u1, u2:
        f2 = (2*(1-u1)*(u2+1)*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/((u2+1)**2+(1-u1)**2)**(3/2)-(2*(1-u1)*(u2+1))/((u2+1)**2+(1-u1)**2)
        update_matrix_value(matrix, u1i, u2i, f2)
        update_matrix_value(matrix, u2i, u1i, f2)
        # matrix[u1i][u2i] += f1
        # matrix[u2i][u1i] += f1

        # p3: Sec-Derivative u2, u2:
        f4 = (2*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/sqrt((u2+1)**2+(1-u1)**2)-(2*(u2+1)**2*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/((u2+1)**2+(1-u1)**2)**(3/2)+(2*(u2+1)**2)/((u2+1)**2+(1-u1)**2)
        update_matrix_value(matrix, u2i, u2i, f4)
        # matrix[u2i][u2i] += f4

    for i in range(30):
        # part 4 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1i = (2 * j + 20) - 1
        u2i = (2 * j) - 1
        u3i = (2 * j + 19) - 1
        u4i = (2 * j - 1) - 1
        u1 = u[u1i]
        u2 = u[u2i]
        u3 = u[u3i]
        u4 = u[u4i]

        # p4: Sec-Derivative u1, u1:
        f1 = (2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)+(2*(-u2+u1+1)**2)/((u3-u4)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)**2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u1i, f1)

        # p4: Sec-Derivative u1, u2:
        f2 = -((2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2))-(2*(-u2+u1+1)**2)/((u3-u4)**2+(-u2+u1+1)**2)+(2*(-u2+u1+1)**2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u2i, f2)
        update_matrix_value(matrix, u2i, u1i, f2)

        # p4: Sec-Derivative u1, u3:
        f3 = (2*(-u2+u1+1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u3i, f3)
        update_matrix_value(matrix, u3i, u1i, f3)

        # p4: Sec-Derivative u1, u4:
        f4 = (2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)-(2*(-u2+u1+1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u1i, u4i, f4)
        update_matrix_value(matrix, u4i, u1i, f4)

        # p4: Sec-Derivative u2, u2:
        f6 = (2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)+(2*(-u2+u1+1)**2)/((u3-u4)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)**2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u2i, u2i, f6)

        # p4: Sec-Derivative u2, u3:
        f7 = (2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)-(2*(-u2+u1+1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u2i, u3i, f7)
        update_matrix_value(matrix, u3i, u2i, f7)

        # p4: Sec-Derivative u2, u4:
        f8 = (2*(-u2+u1+1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u2i, u4i, f8)
        update_matrix_value(matrix, u4i, u2i, f8)

        # p4: Sec-Derivative u3, u3:
        f11 = (2*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)-(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)+(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u3i, u3i, f11)

        # p4: Sec-Derivative u3, u4:
        f12 = -((2*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2))+(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)-(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u3i, u4i, f12)
        update_matrix_value(matrix, u4i, u3i, f12)

        # p4: Sec-Derivative u4, u4:
        f16 = (2*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)-(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)+(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u4i, u4i, f16)

    for i in range(36):
        # part 5 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1i = (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1
        u2i = (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1
        u3i = (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1
        u4i = (2 * j + 2 * floor((j - 1) / 9)) - 1
        u1 = u[u1i]
        u2 = u[u2i]
        u3 = u[u3i]
        u4 = u[u4i]

        # p6: Sec-Derivative u1, u1:
        f1 = (2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)+(2*(-u2+u1+1)**2)/((u3-u4)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)**2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u1i, f1)

        # p6: Sec-Derivative u1, u2:
        f2 = -((2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2))-(2*(-u2+u1+1)**2)/((u3-u4)**2+(-u2+u1+1)**2)+(2*(-u2+u1+1)**2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u2i, f2)
        update_matrix_value(matrix, u2i, u1i, f2)

        # p6: Sec-Derivative u1, u3:
        f3 = (2*(-u2+u1+1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u3i, f3)
        update_matrix_value(matrix, u3i, u1i, f3)

        # p6: Sec-Derivative u1, u4:
        f4 = (2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)-(2*(-u2+u1+1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u1i, u4i, f4)
        update_matrix_value(matrix, u4i, u1i, f4)

        # p6: Sec-Derivative u2, u2:
        f6 = (2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)+(2*(-u2+u1+1)**2)/((u3-u4)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)**2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u2i, u2i, f6)

        # p6: Sec-Derivative u2, u3:
        f7 = (2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)-(2*(-u2+u1+1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u2i, u3i, f7)
        update_matrix_value(matrix, u3i, u2i, f7)

        # p6: Sec-Derivative u2, u4:
        f8 = (2*(-u2+u1+1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u2i, u4i, f8)
        update_matrix_value(matrix, u4i, u2i, f8)

        # p6: Sec-Derivative u3, u3:
        f11 = (2*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)-(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)+(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u3i, u3i, f11)

        # p6: Sec-Derivative u3, u4:
        f12 = -((2*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2))+(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)-(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u3i, u4i, f12)
        update_matrix_value(matrix, u4i, u3i, f12)

        # p6: Sec-Derivative u4, u4:
        f16 = (2*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)-(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4)**2)/((u3-u4)**2+(-u2+u1+1)**2)**(3/2)+(2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u4i, u4i, f16)

    for i in range(27):
        # part 6 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1i = (2 * j + 21 + 2 * floor((j - 1) / 9)) - 1
        u2i = (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1
        u3i = (2 * j + 22 + 2 * floor((j - 1) / 9)) - 1
        u4i = (2 * j + 2 * floor((j - 1) / 9)) - 1
        u1 = u[u1i]
        u2 = u[u2i]
        u3 = u[u3i]
        u4 = u[u4i]

        # p6: Sec-Derivative u1, u1:
        f1 = (2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)+(2*(-u2+u1+1)**2)/((-u4+u3+1)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)**2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u1i, f1)

        # p6: Sec-Derivative u1, u2:
        f2 = -((2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2))-(2*(-u2+u1+1)**2)/((-u4+u3+1)**2+(-u2+u1+1)**2)+(2*(-u2+u1+1)**2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u2i, f2)
        update_matrix_value(matrix, u2i, u1i, f2)

        # p6: Sec-Derivative u1, u3:
        f3 = (2*(-u2+u1+1)*(-u4+u3+1))/((-u4+u3+1)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1))/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u3i, f3)
        update_matrix_value(matrix, u3i, u1i, f3)

        # p6: Sec-Derivative u1, u4:
        f4 = (2*(-u2+u1+1)*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1))/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)-(2*(-u2+u1+1)*(-u4+u3+1))/((-u4+u3+1)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u1i, u4i, f4)
        update_matrix_value(matrix, u4i, u1i, f4)

        # p6: Sec-Derivative u2, u2:
        f6 = (2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)+(2*(-u2+u1+1)**2)/((-u4+u3+1)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)**2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u2i, u2i, f6)

        # p6: Sec-Derivative u2, u3:
        f7 = (2*(-u2+u1+1)*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1))/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)-(2*(-u2+u1+1)*(-u4+u3+1))/((-u4+u3+1)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u2i, u3i, f7)
        update_matrix_value(matrix, u3i, u2i, f7)

        # p6: Sec-Derivative u2, u4:
        f8 = (2*(-u2+u1+1)*(-u4+u3+1))/((-u4+u3+1)**2+(-u2+u1+1)**2)-(2*(-u2+u1+1)*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1))/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)
        update_matrix_value(matrix, u2i, u4i, f8)
        update_matrix_value(matrix, u4i, u2i, f8)

        # p6: Sec-Derivative u3, u3:
        f11 = (2*(-u4+u3+1)**2)/((-u4+u3+1)**2+(-u2+u1+1)**2)-(2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1)**2)/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)+(2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u3i, u3i, f11)

        # p6: Sec-Derivative u3, u4:
        f12 = -((2*(-u4+u3+1)**2)/((-u4+u3+1)**2+(-u2+u1+1)**2))+(2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1)**2)/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)-(2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u3i, u4i, f12)
        update_matrix_value(matrix, u4i, u3i, f12)

        # p6: Sec-Derivative u4, u4:
        f16 = (2*(-u4+u3+1)**2)/((-u4+u3+1)**2+(-u2+u1+1)**2)-(2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1)**2)/((-u4+u3+1)**2+(-u2+u1+1)**2)**(3/2)+(2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)
        update_matrix_value(matrix, u4i, u4i, f16)


    for i in range(27):
        # part 7 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1i = (2 * j + 19 + 2 * floor((j - 1) / 9)) - 1
        u2i = (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1
        u3i = (2 * j + 20 + 2 * floor((j - 1) / 9)) - 1
        u4i = (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1
        u1 = u[u1i]
        u2 = u[u2i]
        u3 = u[u3i]
        u4 = u[u4i]

        # p7: Sec-Derivative u1, u1:
        f1 = (2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2)+(2*(u2-u1+1)**2)/((-u4+u3+1)**2+(u2-u1+1)**2)-(2*(u2-u1+1)**2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u1i, f1)

        # p7: Sec-Derivative u1, u2:
        f2 = -((2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2))-(2*(u2-u1+1)**2)/((-u4+u3+1)**2+(u2-u1+1)**2)+(2*(u2-u1+1)**2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u2i, f2)
        update_matrix_value(matrix, u2i, u1i, f2)

        # p7: Sec-Derivative u1, u3:
        f3 = (2*(u2-u1+1)*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1))/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)-(2*(u2-u1+1)*(-u4+u3+1))/((-u4+u3+1)**2+(u2-u1+1)**2)
        update_matrix_value(matrix, u1i, u3i, f3)
        update_matrix_value(matrix, u3i, u1i, f3)

        # p7: Sec-Derivative u1, u4:
        f4 = (2*(u2-u1+1)*(-u4+u3+1))/((-u4+u3+1)**2+(u2-u1+1)**2)-(2*(u2-u1+1)*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1))/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)
        update_matrix_value(matrix, u1i, u4i, f4)
        update_matrix_value(matrix, u4i, u1i, f4)

        # p7: Sec-Derivative u2, u2:
        f6 = (2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2)+(2*(u2-u1+1)**2)/((-u4+u3+1)**2+(u2-u1+1)**2)-(2*(u2-u1+1)**2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)
        update_matrix_value(matrix, u2i, u2i, f6)

        # p7: Sec-Derivative u2, u3:
        f7 = (2*(u2-u1+1)*(-u4+u3+1))/((-u4+u3+1)**2+(u2-u1+1)**2)-(2*(u2-u1+1)*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1))/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)
        update_matrix_value(matrix, u2i, u3i, f7)
        update_matrix_value(matrix, u3i, u2i, f7)

        # p7: Sec-Derivative u2, u4:
        f8 = (2*(u2-u1+1)*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1))/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)-(2*(u2-u1+1)*(-u4+u3+1))/((-u4+u3+1)**2+(u2-u1+1)**2)
        update_matrix_value(matrix, u2i, u4i, f8)
        update_matrix_value(matrix, u4i, u2i, f8)

        # p7: Sec-Derivative u3, u3:
        f11 = (2*(-u4+u3+1)**2)/((-u4+u3+1)**2+(u2-u1+1)**2)-(2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1)**2)/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)+(2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2)
        update_matrix_value(matrix, u3i, u3i, f11)

        # p7: Sec-Derivative u3, u4:
        f12 = -((2*(-u4+u3+1)**2)/((-u4+u3+1)**2+(u2-u1+1)**2))+(2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1)**2)/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)-(2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2)
        update_matrix_value(matrix, u3i, u4i, f12)
        update_matrix_value(matrix, u4i, u3i, f12)

        # p7: Sec-Derivative u4, u4:
        f16 = (2*(-u4+u3+1)**2)/((-u4+u3+1)**2+(u2-u1+1)**2)-(2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1)**2)/((-u4+u3+1)**2+(u2-u1+1)**2)**(3/2)+(2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2)
        update_matrix_value(matrix, u4i, u4i, f16)
    return matrix


def update_matrix_value(matrix, row, col, current_value):
    matrix[row][col] += current_value


if __name__ == "__main__":
    u80 = np.zeros(80)
    g80 = np.ones(80)
    sec_matrix = sec_derivatives(u80, g80)


    # matrix = model(u80, g80)
    # Create a heatmap
    plt.imshow(sec_matrix, cmap='viridis')  # You can choose different colormaps
    plt.colorbar()
    plt.show()
