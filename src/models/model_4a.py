import numpy as np
from math import sqrt, floor
import math


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
    # matrix = [[] for _ in range(matrix_size)]
    matrix = np.zeros(matrix_size ** 2).reshape((matrix_size, matrix_size))

    for i in range(10):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j - 1) - 1]
        u2 = u[(2 * j) - 1]
        # "p1: Sec-Derivative u1, u1:"
        f1 = (2 * (sqrt((u2 + 1) ** 2 + u1 ** 2) - 1)) / sqrt((u2 + 1) ** 2 + u1 ** 2) \
             - (2 * u1 ** 2 * (sqrt((u2 + 1) ** 2 + u1 ** 2) - 1)) / ((u2 + 1) ** 2 + u1 ** 2) ** (3 / 2) \
             + (2 * u1 ** 2) / ((u2 + 1) ** 2 + u1 ** 2)
        update_matrix_value(matrix, (2 * j - 1) - 1, 0, f1)

        # "p1: Sec-Derivative u1, u2:"
        f2 = (2 * u1 * (u2 + 1)) / ((u2 + 1) ** 2 + u1 ** 2) - (
                2 * u1 * (u2 + 1) * (sqrt((u2 + 1) ** 2 + u1 ** 2) - 1)) / ((u2 + 1) ** 2 + u1 ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j - 1) - 1, 1, f2)

        # "p1: Sec-Derivative u2, u1:"
        f3 = (2 * u1 * (u2 + 1)) / ((u2 + 1) ** 2 + u1 ** 2) - (
                2 * u1 * (u2 + 1) * (sqrt((u2 + 1) ** 2 + u1 ** 2) - 1)) / ((u2 + 1) ** 2 + u1 ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j) - 1, 0, f3)

        # "p1: Sec-Derivative u2, u2:"
        f4 = (2 * (sqrt((u2 + 1) ** 2 + u1 ** 2) - 1)) / sqrt((u2 + 1) ** 2 + u1 ** 2) \
             - (2 * (u2 + 1) ** 2 * (sqrt((u2 + 1) ** 2 + u1 ** 2) - 1)) / ((u2 + 1) ** 2 + u1 ** 2) ** (3 / 2) \
             + (2 * (u2 + 1) ** 2) / ((u2 + 1) ** 2 + u1 ** 2)
        update_matrix_value(matrix, (2 * j) - 1, 1, f3)

    for i in range(9):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 1) - 1]
        u2 = u[(2 * j + 2) - 1]
        # "p2: Sec-Derivative u1, u1:"
        f1 = (2 * (sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) - sqrt(2))) / sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) \
             - (2 * (u1 + 1) ** 2 * (sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) - sqrt(2))) / (
                     (u2 + 1) ** 2 + (u1 + 1) ** 2) ** (3 / 2) \
             + (2 * (u1 + 1) ** 2) / ((u2 + 1) ** 2 + (u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 1) - 1, 0, f1)

        # "p2: Sec-Derivative u1, u2:"
        f2 = (2 * (u1 + 1) * (u2 + 1)) / ((u2 + 1) ** 2 + (u1 + 1) ** 2) \
             - (2 * (u1 + 1) * (u2 + 1) * (sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) - sqrt(2))) / (
                     (u2 + 1) ** 2 + (u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 1) - 1, 1, f2)

        # "p2: Sec-Derivative u2, u1:"
        f3 = (2 * (u1 + 1) * (u2 + 1)) / ((u2 + 1) ** 2 + (u1 + 1) ** 2) \
             - (2 * (u1 + 1) * (u2 + 1) * (sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) - sqrt(2))) / (
                     (u2 + 1) ** 2 + (u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 2) - 1, 0, f3)

        # "p2: Sec-Derivative u2, u2:"
        f4 = (2 * (sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) - sqrt(2))) / sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) \
        - (2 * (u2 + 1) ** 2 * (sqrt((u2 + 1) ** 2 + (u1 + 1) ** 2) - sqrt(2))) / ((u2 + 1) ** 2 + (u1 + 1) ** 2) ** (
                3 / 2) \
        + (2 * (u2 + 1) ** 2) / ((u2 + 1) ** 2 + (u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2) - 1, 1, f4)

    for i in range(9):
        # part 3 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j - 1) - 1]
        u2 = u[(2 * j) - 1]

        # "p3: Sec-Derivative u1, u1:"
        f1 = (2 * (sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) - sqrt(2))) / sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) \
             - (2 * (1 - u1) ** 2 * (sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) - sqrt(2))) / (
                     (u2 + 1) ** 2 + (1 - u1) ** 2) ** (3 / 2) \
             + (2 * (1 - u1) ** 2) / ((u2 + 1) ** 2 + (1 - u1) ** 2)
        update_matrix_value(matrix, (2 * j - 1) - 1, 0, f1)

        # "p3: Sec-Derivative u1, u2:"
        f2 = (2 * (1 - u1) * (u2 + 1) * (sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) - sqrt(2))) / (
                (u2 + 1) ** 2 + (1 - u1) ** 2) ** (3 / 2) \
             - (2 * (1 - u1) * (u2 + 1)) / ((u2 + 1) ** 2 + (1 - u1) ** 2)
        update_matrix_value(matrix, (2 * j - 1) - 1, 1, f2)

        # "p3: Sec-Derivative u2, u1:"
        f3 = (2 * (1 - u1) * (u2 + 1) * (sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) - sqrt(2))) / (
                (u2 + 1) ** 2 + (1 - u1) ** 2) ** (3 / 2) \
             - (2 * (1 - u1) * (u2 + 1)) / ((u2 + 1) ** 2 + (1 - u1) ** 2)
        update_matrix_value(matrix, (2 * j) - 1, 0, f3)

        # "p3: Sec-Derivative u2, u2:"
        f4 = (2 * (sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) - sqrt(2))) / sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) \
             - (2 * (u2 + 1) ** 2 * (sqrt((u2 + 1) ** 2 + (1 - u1) ** 2) - sqrt(2))) / (
                     (u2 + 1) ** 2 + (1 - u1) ** 2) ** (3 / 2) \
             + (2 * (u2 + 1) ** 2) / ((u2 + 1) ** 2 + (1 - u1) ** 2)
        update_matrix_value(matrix, (2 * j) - 1, 1, f4)

    for i in range(30):
        # part 4 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 20) - 1]
        u2 = u[(2 * j) - 1]
        u3 = u[(2 * j + 19) - 1]
        u4 = u[(2 * j - 1) - 1]

        # "p4: Sec-Derivative u1, u1:"
        f1 = (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) ** 2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / (
                     (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 20) - 1, 0, f1)

        # "p4: Sec-Derivative u1, u2:"
        f2 = -((2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)) \
             - (2 * (-u2 + u1 + 1) ** 2) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / (
                     (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j) - 1, 1, f2)

        # "p4: Sec-Derivative u1, u3:"
        f3 = (2 * (-u2 + u1 + 1) * (u3 - u4)) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / (
                     (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 19) - 1, 2, f3)

        # "p4: Sec-Derivative u1, u4:"
        f4 = (2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / (
                (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
             - (2 * (-u2 + u1 + 1) * (u3 - u4)) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j - 1) - 1, 3, f4)

        # "p4: Sec-Derivative u2, u1:"
        f5 = -((2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)) \
             - (2 * (-u2 + u1 + 1) ** 2) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / (
                     (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 20) - 1, 0, f5)

        # "p4: Sec-Derivative u2, u2:"
        f6 = (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) ** 2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / (
                     (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j) - 1, 1, f6)

        # "p4: Sec-Derivative u2, u3:"
        f7 = (2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / (
                (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
             - (2 * (-u2 + u1 + 1) * (u3 - u4)) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 19) - 1, 2, f7)

        # "p4: Sec-Derivative u2, u4:"
        f8 = (2 * (-u2 + u1 + 1) * (u3 - u4)) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / (
                     (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j - 1) - 1, 3, f8)

        # "p4: Sec-Derivative u3, u1:"
        f9 = (2 * (-u2 + u1 + 1) * (u3 - u4)) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / (
                     (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 20) - 1, 0, f9)

        # "p4: Sec-Derivative u3, u2:"
        f10 = (2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / (
                (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (-u2 + u1 + 1) * (u3 - u4)) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j) - 1, 1, f10)

        # "p4: Sec-Derivative u3, u3:"
        f11 = (2 * (u3 - u4) ** 2) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
              - (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4) ** 2) / (
                      (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              + (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 19) - 1, 2, f11)

        # "p4: Sec-Derivative u3, u4:"
        f12 = -((2 * (u3 - u4) ** 2) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)) \
              + (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4) ** 2) / (
                      (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j - 1) - 1, 3, f12)

        # "p4: Sec-Derivative u4, u1:"
        f13 = (2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / (
                (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (-u2 + u1 + 1) * (u3 - u4)) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 20) - 1, 0, f13)

        # "p4: Sec-Derivative u4, u2:"
        f14 = (2 * (-u2 + u1 + 1) * (u3 - u4)) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
              - (2 * (-u2 + u1 + 1) * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4)) / (
                      (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j) - 1, 1, f14)

        # "p4: Sec-Derivative u4, u3:"
        f15 = -((2 * (u3 - u4) ** 2) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)) \
              + (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4) ** 2) / (
                      (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 19) - 1, 2, f15)

        # "p4: Sec-Derivative u4, u4:"
        f16 = (2 * (u3 - u4) ** 2) / ((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) \
              - (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1) * (u3 - u4) ** 2) / (
                      (u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              + (2 * (sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2) - 1)) / sqrt((u3 - u4) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j - 1) - 1, 3, f16)

    for i in range(36):
        # part 5 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 1 + 2 * floor((j - 1) / 9)) - 1]
        u2 = u[(2 * j - 1 + 2 * floor((j - 1) / 9)) - 1]
        u3 = u[(2 * j + 2 + 2 * floor((j - 1) / 9)) - 1]
        u4 = u[(2 * j + 2 * floor((j - 1) / 9)) - 1]

        # "p6: Sec-Derivative u1, u1:"
        f1 = (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1, 0, f1)

        # "p6: Sec-Derivative u1, u2:"
        f2 = -((2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)) \
             - (2 * (-u2 + u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1, 1, f2)

        # "p6: Sec-Derivative u1, u3:"
        f3 = (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2
                                           + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1, 2, f3)

        # "p6: Sec-Derivative u1, u4:"
        f4 = (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                                    - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                     3 / 2) \
             - (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1, 3, f4)

        # "p6: Sec-Derivative u2, u1:"
        f5 = -((2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)) \
             - (2 * (-u2 + u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1, 0, f5)

        # "p6: Sec-Derivative u2, u2:"
        f6 = (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1, 1, f6)

        # "p6: Sec-Derivative u2, u3:"
        f7 = (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                                    - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                     3 / 2) \
             - (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1, 2, f7)

        # "p6: Sec-Derivative u2, u4:"
        f8 = (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                                      - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                     3 / 2)
        update_matrix_value(matrix, (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1, 3, f8)

        # p6: Sec-Derivative u3, u1:"
        f9 = (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                                      - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                     3 / 2)
        update_matrix_value(matrix, (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1, 0, f9)

        # "p6: Sec-Derivative u3, u2:"
        f10 = (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                      3 / 2) \
              - (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1, 1, f10)

        # "p6: Sec-Derivative u3, u3:"
        f11 = (2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        -(2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
               - sqrt(2)) * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
        + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1, 2, f11)

        # "p6: Sec-Derivative u3, u4:"
        f12 = -((2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                      - sqrt(2)) * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1, 3, f12)

        # "p6: Sec-Derivative u4, u1:"
        f13 = (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
              / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 * floor((j - 1) / 9)) - 1, 0, f13)

        # "p6: Sec-Derivative u4, u2:"
        f14 = (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
              - (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
              / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 2 * floor((j - 1) / 9)) - 1, 1, f14)

        # "p6: Sec-Derivative u4, u3:"
        f15 = -((2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1) ** 2) \
              / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) \
              / sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 * floor((j - 1) / 9)) - 1, 2, f15)

        # "p6: Sec-Derivative u4, u4:"
        f16 = (2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1) ** 2) \
              / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) \
              / sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 * floor((j - 1) / 9)) - 1, 3, f16)

    for i in range(27):
        # part 6 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 21 + 2 * floor((j - 1) / 9)) - 1]
        u2 = u[(2 * j - 1 + 2 * floor((j - 1) / 9)) - 1]
        u3 = u[(2 * j + 22 + 2 * floor((j - 1) / 9)) - 1]
        u4 = u[(2 * j + 2 * floor((j - 1) / 9)) - 1]

        # "p6: Sec-Derivative u1, u1:"
        f1 = (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 21 + 2 * floor((j - 1) / 9)) - 1, 0, f1)

        # "p6: Sec-Derivative u1, u2:"
        f2 = -((2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)) \
             - (2 * (-u2 + u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 21 + 2 * floor((j - 1) / 9)) - 1, 1, f2)

        # "p6: Sec-Derivative u1, u3:"
        f3 = (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2
                                           + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 21 + 2 * floor((j - 1) / 9)) - 1, 2, f3)

        # "p6: Sec-Derivative u1, u4:"
        f4 = (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                                    - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                     3 / 2) \
             - (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 21 + 2 * floor((j - 1) / 9)) - 1, 3, f4)

        # "p6: Sec-Derivative u2, u1:"
        f5 = -((2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)) \
             - (2 * (-u2 + u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1, 0, f5)

        # "p6: Sec-Derivative u2, u2:"
        f6 = (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             + (2 * (-u2 + u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / (
                     (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1, 1, f6)

        # "p6: Sec-Derivative u2, u3:"
        f7 = (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                                    - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                     3 / 2) \
             - (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1, 2, f7)

        # "p6: Sec-Derivative u2, u4:"
        f8 = (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                                      - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                     3 / 2)
        update_matrix_value(matrix, (2 * j - 1 + 2 * floor((j - 1) / 9)) - 1, 3, f8)

        # p6: Sec-Derivative u3, u1:"
        f9 = (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
             - (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                                      - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                     3 / 2)
        update_matrix_value(matrix, (2 * j + 22 + 2 * floor((j - 1) / 9)) - 1, 0, f9)

        # "p6: Sec-Derivative u3, u2:"
        f10 = (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                                     - sqrt(2)) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (
                      3 / 2) \
              - (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 22 + 2 * floor((j - 1) / 9)) - 1, 1, f10)

        # "p6: Sec-Derivative u3, u3:"
        f11 = (2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        -(2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
               - sqrt(2)) * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
        + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 22 + 2 * floor((j - 1) / 9)) - 1, 2, f11)

        # "p6: Sec-Derivative u3, u4:"
        f12 = -((2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
                      - sqrt(2)) * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 22 + 2 * floor((j - 1) / 9)) - 1, 3, f12)

        # "p6: Sec-Derivative u4, u1:"
        f13 = (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
              / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 * floor((j - 1) / 9)) - 1, 0, f13)

        # "p6: Sec-Derivative u4, u2:"
        f14 = (2 * (-u2 + u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
              - (2 * (-u2 + u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
              / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 2 * floor((j - 1) / 9)) - 1, 1, f14)

        # "p6: Sec-Derivative u4, u3:"
        f15 = -((2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1) ** 2) \
              / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) \
              / sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 * floor((j - 1) / 9)) - 1, 2, f15)

        # "p6: Sec-Derivative u4, u4:"
        f16 = (2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1) ** 2) \
              / ((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) ** (3 / 2) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2) - sqrt(2))) \
              / sqrt((-u4 + u3 + 1) ** 2 + (-u2 + u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 * floor((j - 1) / 9)) - 1, 3, f16)

    for i in range(27):
        # part 7 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2 * j + 19 + 2 * floor((j - 1) / 9)) - 1]
        u2 = u[(2 * j + 1 + 2 * floor((j - 1) / 9)) - 1]
        u3 = u[(2 * j + 20 + 2 * floor((j - 1) / 9)) - 1]
        u4 = u[(2 * j + 2 + 2 * floor((j - 1) / 9)) - 1]

        # "p7: Sec-Derivative u1, u1:"
        f1 = (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
             + (2 * (u2 - u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
             - (2 * (u2 - u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) \
             / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 19 + 2 * floor((j - 1) / 9)) - 1, 3, f1)

        # "p7: Sec-Derivative u1, u2:"
        f2 = -((2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)))
               / sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)) \
             - (2 * (u2 - u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
             + (2 * (u2 - u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) \
             / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 19 + 2 * floor((j - 1) / 9)) - 1, 3, f2)

        # "p7: Sec-Derivative u1, u3:"
        f3 = (2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
             / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2) \
             - (2 * (u2 - u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 19 + 2 * floor((j - 1) / 9)) - 1, 3, f3)

        # "p7: Sec-Derivative u1, u4:"
        f4 = (2 * (u2 - u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
             - (2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
             / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 19 + 2 * floor((j - 1) / 9)) - 1, 3, f4)

        # "p7: Sec-Derivative u2, u1:"
        f5 = -((2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)))
               / sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)) \
             - (2 * (u2 - u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
             + (2 * (u2 - u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) \
             / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1, 3, f5)

        # "p7: Sec-Derivative u2, u2:"
        f6 = (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
             + (2 * (u2 - u1 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
             - (2 * (u2 - u1 + 1) ** 2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) \
             / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1, 3, f6)

        # "p7: Sec-Derivative u2, u3:"
        f7 = (2 * (u2 - u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
             - (2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
             / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1, 3, f7)

        # "p7: Sec-Derivative u2, u4:"
        f8 = (2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
             / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2) \
             - (2 * (u2 - u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 1 + 2 * floor((j - 1) / 9)) - 1, 3, f8)

        # "p7: Sec-Derivative u3, u1:"
        f9 = (2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
             / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2) \
             - (2 * (u2 - u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 20 + 2 * floor((j - 1) / 9)) - 1, 3, f9)

        # "p7: Sec-Derivative u3, u2:"
        f10 = (2 * (u2 - u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
              - (2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
              / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 20 + 2 * floor((j - 1) / 9)) - 1, 3, f10)

        # "p7: Sec-Derivative u3, u3:"
        f11 = (2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1) ** 2) \
              / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 20 + 2 * floor((j - 1) / 9)) - 1, 3, f11)

        # "p7: Sec-Derivative u3, u4:"
        f12 = -((2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1) ** 2) \
              / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 20 + 2 * floor((j - 1) / 9)) - 1, 3, f12)

        # "p7: Sec-Derivative u4, u1:"
        f13 = (2 * (u2 - u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
              - (2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
              / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2)
        update_matrix_value(matrix, (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1, 3, f13)

        # "p7: Sec-Derivative u4, u2:"
        f14 = (2 * (u2 - u1 + 1) * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1)) \
              / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (u2 - u1 + 1) * (-u4 + u3 + 1)) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1, 3, f14)

        # "p7: Sec-Derivative u4, u3:"
        f15 = -((2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1) ** 2) \
              / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1, 3, f15)

        # "p7: Sec-Derivative u4, u4:"
        f16 = (2 * (-u4 + u3 + 1) ** 2) / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) \
              - (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2)) * (-u4 + u3 + 1) ** 2) \
              / ((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) ** (3 / 2) \
              + (2 * (sqrt((-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2) - sqrt(2))) / sqrt(
            (-u4 + u3 + 1) ** 2 + (u2 - u1 + 1) ** 2)
        update_matrix_value(matrix, (2 * j + 2 + 2 * floor((j - 1) / 9)) - 1, 3, f16)

    # numpy_array = np.array(matrix)
    # print(matrix)
    return matrix


def update_matrix_value(matrix, row, col, current_value):
    matrix[row][col] += current_value
    # try:
    #     matrix[row][col] += current_value
    # except:
    #     pass
    # finally:
    #     matrix[row].append(current_value)


if __name__ == "__main__":
    u80 = np.ones(80)
    g80 = np.ones(80)
    print(sec_derivatives(u80, g80))
