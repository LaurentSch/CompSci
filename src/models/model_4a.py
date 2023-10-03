import numpy as np
from math import sqrt, floor
import math


def model_4a(u, g):
    m = -np.dot(g.T, u)

    for i in range(10):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j-1]
        u2 = u[2*j]
        m += (sqrt(u1**2 + (1 + u2)**2) - 1)**2

    for i in range(9):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+1]
        u2 = u[2*j+2]
        m += (sqrt((1 + u1)**2 + (1 + u2)**2) - sqrt(2))**2


    for i in range(9):
        # part 3 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j-1]
        u2 = u[2*j]
        m += (sqrt((1 - u1)**2 + (1 + u2)**2) - sqrt(2))**2

    for i in range(30):
        # part 4 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+20]
        u2 = u[2*j]
        u3 = u[2*j+19]
        u4 = u[2*j-1]
        m += (sqrt((1 + u1 - u2)**2 + (u3 - u4)**2) - 1)**2

    for i in range(58):
        # part 5 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+1+2*floor((j-1)/9)]
        u2 = u[2*j-1+2*floor((j-1)/9)]
        u3 = u[2*j+2+2*floor((j-1)/9)]
        u4 = u[2*j+2*floor((j-1)/9)]
        m += (sqrt((1 + u1 - u2)**2 + (u3 - u4)**2) - 1)**2

    for i in range(94):
        # part 6 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+21+2*floor((j-1)/9)]
        u2 = u[2*j-1+2*floor((j-1)/9)]
        u3 = u[2*j+22+2*floor((j-1)/9)]
        u4 = u[2*j+2*floor((j-1)/9)]
        m += (sqrt((1 + u1 - u2)**2 + (1 + u3 - u4)**2) - sqrt(2))**2

    for i in range(121):
        # part 7 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+19+2*floor((j-1)/9)]
        u2 = u[2*j+1+2*floor((j-1)/9)]
        u3 = u[2*j+20+2*floor((j-1)/9)]
        u4 = u[2*j+2+2*floor((j-1)/9)]
        m += (sqrt((1 - u1 + u2)**2 + (1 + u3 - u4)**2) - sqrt(2))**2
    return m


# g = np.zeros(80)
def model_4a_derivative(u, g):
    f_new = - g

    for i in range(10):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j-1]
        u2 = u[2*j]
        # derivative for u[2*i-1]
        f1 = (2*u1*(sqrt((u2+1)**2+u1**2)-1))/sqrt((u2+1)**2+u1**2)
        # derivative for u[2*i]
        f2 = (2*(u2+1)*(sqrt((u2+1)**2+u1**2)-1))/sqrt((u2+1)**2+u1**2)
        # Add results to the gradient
        f_new[2*j-1] += f1
        f_new[2*j] += f2

    for i in range(9):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+1]
        u2 = u[2*j+2]
        # derivative for u[2*i-1]
        f1 = (2*(u1+1)*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/sqrt((u2+1)**2+(u1+1)**2)
        # derivative for u[2*i]
        f2 = (2*(u2+1)*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/sqrt((u2+1)**2+(u1+1)**2)
        # Add results to the gradient
        f_new[2*j+1] += f1
        f_new[2*j+2] += f2

    for i in range(9):
        # part 3 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j-1]
        u2 = u[2*j]
        # derivative for u[2*i-1]
        f1 = -((2*(1-u1)*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/sqrt((u2+1)**2+(1-u1)**2))
        # derivative for u[2*i]
        f2 = (2*(u2+1)*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/sqrt((u2+1)**2+(1-u1)**2)
        # Add results to the gradient
        f_new[2*j-1] += f1
        f_new[2*j] += f2

    for i in range(30):
        # part 4 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+20]
        u2 = u[2*j]
        u3 = u[2*j+19]
        u4 = u[2*j-1]
        # derivative for u[2*i-1]
        f1 = (2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        # derivative for u[2*i]
        f2 = -((2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        f3 = (2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        f4 = -((2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        # Add results to the gradient
        f_new[2*j+20] += f1
        f_new[2*j] += f2
        f_new[2*j+19] += f3
        f_new[2*j-1] += f4

    for i in range(58):
        # part 5 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+1+2*floor((j-1)/9)]
        u2 = u[2*j-1+2*floor((j-1)/9)]
        u3 = u[2*j+2+2*floor((j-1)/9)]
        u4 = u[2*j+2*floor((j-1)/9)]
        # derivative for u[2*i-1]
        f1 = (2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        # derivative for u[2*i]
        f2 = -((2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        f3 = (2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        f4 = -((2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        # Add results to the gradient
        f_new[2*j+1+2*floor((j-1)/9)] += f1
        f_new[2*j-1+2*floor((j-1)/9)] += f2
        f_new[2*j+2+2*floor((j-1)/9)] += f3
        f_new[2*j+2*floor((j-1)/9)] += f4

    for i in range(94):
        # part 6 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+21+2*floor((j-1)/9)]
        u2 = u[2*j-1+2*floor((j-1)/9)]
        u3 = u[2*j+22+2*floor((j-1)/9)]
        u4 = u[2*j+2*floor((j-1)/9)]
        # derivative for u[2*i-1]
        f1 = (2*(-u2+u1+1)*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)
        # derivative for u[2*i]
        f2 = -((2*(-u2+u1+1)*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2))
        f3 = (2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)
        f4 = -((2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2))
        # Add results to the gradient
        f_new[2*j+21+2*floor((j-1)/9)] += f1
        f_new[2*j-1+2*floor((j-1)/9)] += f2
        f_new[2*j+22+2*floor((j-1)/9)] += f3
        f_new[2*j+2*floor((j-1)/9)] += f4

    for i in range(121):
        # part 7 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i+1
        u1 = u[2*j+19+2*floor((j-1)/9)]
        u2 = u[2*j+1+2*floor((j-1)/9)]
        u3 = u[2*j+20+2*floor((j-1)/9)]
        u4 = u[2*j+2+2*floor((j-1)/9)]
        # derivative for u[2*i-1]
        f1 = -((2*(u2-u1+1)*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2))
        # derivative for u[2*i]
        f2 = (2*(u2-u1+1)*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2)
        f3 = (2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2)
        f4 = -((2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2))
        # Add results to the gradient
        f_new[2*j+19+2*floor((j-1)/9)] += f1
        f_new[2*j+1+2*floor((j-1)/9)] += f2
        f_new[2*j+20+2*floor((j-1)/9)] += f3
        f_new[2*j+2+2*floor((j-1)/9)] += f4
    return f_new
