import numpy as np
from math import sqrt, floor
import math


def model(u, g):
    m = -np.dot(g, u)

    for i in range(10):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j-1)-1]
        u2 = u[(2*j)-1]
        m += (sqrt(u1**2 + (1 + u2)**2) - 1)**2

    for i in range(9):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j+1)-1]
        u2 = u[(2*j+2)-1]
        m += (sqrt((1 + u1)**2 + (1 + u2)**2) - sqrt(2))**2

    for i in range(9):
        # part 3 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j-1)-1]
        u2 = u[(2*j)-1]
        m += (sqrt((1 - u1)**2 + (1 + u2)**2) - sqrt(2))**2

    for i in range(30):
        # part 4 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j+20)-1]
        u2 = u[(2*j)-1]
        u3 = u[(2*j+19)-1]
        u4 = u[(2*j-1)-1]
        m += (sqrt((1 + u1 - u2)**2 + (u3 - u4)**2) - 1)**2

    for i in range(36):
        # part 5 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i
        u1 = u[(2*j+1+2*floor((j-1)/9))-1]
        u2 = u[(2*j-1+2*floor((j-1)/9))-1]
        u3 = u[(2*j+2+2*floor((j-1)/9))-1]
        u4 = u[(2*j+2*floor((j-1)/9))-1]
        m += (sqrt((1 + u1 - u2)**2 + (u3 - u4)**2) - 1)**2

    for i in range(27):
        # part 6 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j+21+2*floor((j-1)/9))-1]
        u2 = u[(2*j-1+2*floor((j-1)/9))-1]
        u3 = u[(2*j+22+2*floor((j-1)/9))-1]
        u4 = u[(2*j+2*floor((j-1)/9))-1]
        m += (sqrt((1 + u1 - u2)**2 + (1 + u3 - u4)**2) - sqrt(2))**2

    for i in range(27):
        # part 7 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j+19+2*floor((j-1)/9))-1]
        u2 = u[(2*j+1+2*floor((j-1)/9))-1]
        u3 = u[(2*j+20+2*floor((j-1)/9))-1]
        u4 = u[(2*j+2+2*floor((j-1)/9))-1]
        m += (sqrt((1 - u1 + u2)**2 + (1 + u3 - u4)**2) - sqrt(2))**2

    return m


# g = np.zeros(80)
def derivatives(u, g):
    f_new = - g
    for i in range(10):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j-1)-1]
        u2 = u[(2*j)-1]
        # derivative for u[2*i-1]
        f1 = (2*u1*(sqrt((u2+1)**2+u1**2)-1))/sqrt((u2+1)**2+u1**2)
        # derivative for u[2*i]
        f2 = (2*(u2+1)*(sqrt((u2+1)**2+u1**2)-1))/sqrt((u2+1)**2+u1**2)
        # Add results to the gradient
        f_new[(2*j-1)-1] += f1
        f_new[(2*j)-1] += f2

    for i in range(9):
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j+1)-1]
        u2 = u[(2*j+2)-1]
        # derivative for u[2*i-1]
        f1 = (2*(u1+1)*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/sqrt((u2+1)**2+(u1+1)**2)
        # derivative for u[2*i]
        f2 = (2*(u2+1)*(sqrt((u2+1)**2+(u1+1)**2)-sqrt(2)))/sqrt((u2+1)**2+(u1+1)**2)
        # Add results to the gradient
        f_new[(2*j+1)-1] += f1
        f_new[(2*j+2)-1] += f2

    for i in range(9):
        # part 3 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j-1)-1]
        u2 = u[(2*j)-1]
        # derivative for u[2*i-1]
        f1 = -((2*(1-u1)*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/sqrt((u2+1)**2+(1-u1)**2))
        # derivative for u[2*i]
        f2 = (2*(u2+1)*(sqrt((u2+1)**2+(1-u1)**2)-sqrt(2)))/sqrt((u2+1)**2+(1-u1)**2)
        # Add results to the gradient
        f_new[(2*j-1)-1] += f1
        f_new[(2*j)-1] += f2

    for i in range(30):
        # part 4 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j+20)-1]
        u2 = u[(2*j)-1]
        u3 = u[(2*j+19)-1]
        u4 = u[(2*j-1)-1]
        # derivative for u[2*i-1]
        f1 = (2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        # derivative for u[2*i]
        f2 = -((2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        f3 = (2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        f4 = -((2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        # Add results to the gradient
        f_new[(2*j+20)-1] += f1
        f_new[(2*j)-1] += f2
        f_new[(2*j+19)-1] += f3
        f_new[(2*j-1)-1] += f4

    for i in range(36):
        # part 5 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j+1+2*floor((j-1)/9))-1]
        u2 = u[(2*j-1+2*floor((j-1)/9))-1]
        u3 = u[(2*j+2+2*floor((j-1)/9))-1]
        u4 = u[(2*j+2*floor((j-1)/9))-1]
        # derivative for u[2*i-1]
        f1 = (2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        # derivative for u[2*i]
        f2 = -((2*(-u2+u1+1)*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        f3 = (2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2)
        f4 = -((2*(sqrt((u3-u4)**2+(-u2+u1+1)**2)-1)*(u3-u4))/sqrt((u3-u4)**2+(-u2+u1+1)**2))
        # Add results to the gradient
        f_new[(2*j+1+2*floor((j-1)/9))-1] += f1
        f_new[(2*j-1+2*floor((j-1)/9))-1] += f2
        f_new[(2*j+2+2*floor((j-1)/9))-1] += f3
        f_new[(2*j+2*floor((j-1)/9))-1] += f4

    for i in range(27):
        # part 6 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j+21+2*floor((j-1)/9))-1]
        u2 = u[(2*j-1+2*floor((j-1)/9))-1]
        u3 = u[(2*j+22+2*floor((j-1)/9))-1]
        u4 = u[(2*j+2*floor((j-1)/9))-1]
        # derivative for u[2*i-1]
        f1 = (2*(-u2+u1+1)*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)
        # derivative for u[2*i]
        f2 = -((2*(-u2+u1+1)*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2))
        f3 = (2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)
        f4 = -((2*(sqrt((-u4+u3+1)**2+(-u2+u1+1)**2)-sqrt(2))*(-u4+u3+1))/sqrt((-u4+u3+1)**2+(-u2+u1+1)**2))
        # Add results to the gradient
        f_new[(2*j+21+2*floor((j-1)/9))-1] += f1
        f_new[(2*j-1+2*floor((j-1)/9))-1] += f2
        f_new[(2*j+22+2*floor((j-1)/9))-1] += f3
        f_new[(2*j+2*floor((j-1)/9))-1] += f4

    for i in range(27):
        # part 7 loop
        # fix so that it goes from 1 to 10 instead of 0 to 9
        j = i + 1
        u1 = u[(2*j+19+2*floor((j-1)/9))-1]
        u2 = u[(2*j+1+2*floor((j-1)/9))-1]
        u3 = u[(2*j+20+2*floor((j-1)/9))-1]
        u4 = u[(2*j+2+2*floor((j-1)/9))-1]
        # derivative for u[2*i-1]
        f1 = -((2*(u2-u1+1)*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2))
        # derivative for u[2*i]
        f2 = (2*(u2-u1+1)*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2)))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2)
        f3 = (2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2)
        f4 = -((2*(sqrt((-u4+u3+1)**2+(u2-u1+1)**2)-sqrt(2))*(-u4+u3+1))/sqrt((-u4+u3+1)**2+(u2-u1+1)**2))
        # Add results to the gradient
        f_new[(2*j+19+2*floor((j-1)/9))-1] += f1
        f_new[(2*j+1+2*floor((j-1)/9))-1] += f2
        f_new[(2*j+20+2*floor((j-1)/9))-1] += f3
        f_new[(2*j+2+2*floor((j-1)/9))-1] += f4
    return f_new


def sec_derivatives(u, g):
    p1: Sec-Derivative u1, u1:
    (%o51) "p1: Sec-Derivative u1, u1:"
    (%o52) (2*(sqrt((u2+1)^2+u1^2)-1))/sqrt((u2+1)^2+u1^2)
    -(2*u1^2*(sqrt((u2+1)^2+u1^2)-1))/((u2+1)^2+u1^2)^(3/2)
    +(2*u1^2)/((u2+1)^2+u1^2)
    p1: Sec-Derivative u1, u2:
    (%o53) "p1: Sec-Derivative u1, u2:"
    (%o54) (2*u1*(u2+1))/((u2+1)^2+u1^2)-(2*u1*(u2+1)*(sqrt((u2+1)^2+u1^2)-1))
                     /((u2+1)^2+u1^2)^(3/2)
    p1: Sec-Derivative u2, u1:
    (%o55) "p1: Sec-Derivative u2, u1:"
    (%o56) (2*u1*(u2+1))/((u2+1)^2+u1^2)-(2*u1*(u2+1)*(sqrt((u2+1)^2+u1^2)-1))
                     /((u2+1)^2+u1^2)^(3/2)
    p2: Sec-Derivative u2, u2:
    (%o57) "p2: Sec-Derivative u2, u2:"
    (%o58) (2*(sqrt((u2+1)^2+u1^2)-1))/sqrt((u2+1)^2+u1^2)
    -(2*(u2+1)^2*(sqrt((u2+1)^2+u1^2)-1))/((u2+1)^2+u1^2)^(3/2)
    +(2*(u2+1)^2)/((u2+1)^2+u1^2)
    p2: Sec-Derivative u1, u1:
    (%o59) "p2: Sec-Derivative u1, u1:"
    (%o60) (2*(sqrt((u2+1)^2+(u1+1)^2)-sqrt(2)))/sqrt((u2+1)^2+(u1+1)^2)
    -(2*(u1+1)^2*(sqrt((u2+1)^2+(u1+1)^2)-sqrt(2)))/((u2+1)^2+(u1+1)^2)^(3/2)
    +(2*(u1+1)^2)/((u2+1)^2+(u1+1)^2)
    p1: Sec-Derivative u1, u2:
    (%o61) "p1: Sec-Derivative u1, u2:"
    (%o62) (2*(u1+1)*(u2+1))/((u2+1)^2+(u1+1)^2)
    -(2*(u1+1)*(u2+1)*(sqrt((u2+1)^2+(u1+1)^2)-sqrt(2)))
    /((u2+1)^2+(u1+1)^2)^(3/2)
    p2: Sec-Derivative u2, u1:
    (%o63) "p2: Sec-Derivative u2, u1:"
    (%o64) (2*(u1+1)*(u2+1))/((u2+1)^2+(u1+1)^2)
    -(2*(u1+1)*(u2+1)*(sqrt((u2+1)^2+(u1+1)^2)-sqrt(2)))
    /((u2+1)^2+(u1+1)^2)^(3/2)
    p2: Sec-Derivative u2, u2:
    (%o65) "p2: Sec-Derivative u2, u2:"
    (%o66) (2*(sqrt((u2+1)^2+(u1+1)^2)-sqrt(2)))/sqrt((u2+1)^2+(u1+1)^2)
    -(2*(u2+1)^2*(sqrt((u2+1)^2+(u1+1)^2)-sqrt(2)))/((u2+1)^2+(u1+1)^2)^(3/2)
    +(2*(u2+1)^2)/((u2+1)^2+(u1+1)^2)
    p3: Sec-Derivative u1, u1:
    (%o67) "p3: Sec-Derivative u1, u1:"
    (%o68) (2*(sqrt((u2+1)^2+(1-u1)^2)-sqrt(2)))/sqrt((u2+1)^2+(1-u1)^2)
    -(2*(1-u1)^2*(sqrt((u2+1)^2+(1-u1)^2)-sqrt(2)))/((u2+1)^2+(1-u1)^2)^(3/2)
    +(2*(1-u1)^2)/((u2+1)^2+(1-u1)^2)
    p3: Sec-Derivative u1, u2:
    (%o69) "p3: Sec-Derivative u1, u2:"
    (%o70) (2*(1-u1)*(u2+1)*(sqrt((u2+1)^2+(1-u1)^2)-sqrt(2)))/((u2+1)^2+(1-u1)^2)
                                           ^(3/2)
    -(2*(1-u1)*(u2+1))/((u2+1)^2+(1-u1)^2)
    p3: Sec-Derivative u2, u1:
    (%o71) "p3: Sec-Derivative u2, u1:"
    (%o72) (2*(1-u1)*(u2+1)*(sqrt((u2+1)^2+(1-u1)^2)-sqrt(2)))/((u2+1)^2+(1-u1)^2)
                                           ^(3/2)
    -(2*(1-u1)*(u2+1))/((u2+1)^2+(1-u1)^2)
    p3: Sec-Derivative u2, u2:
    (%o73) "p3: Sec-Derivative u2, u2:"
    (%o74) (2*(sqrt((u2+1)^2+(1-u1)^2)-sqrt(2)))/sqrt((u2+1)^2+(1-u1)^2)
    -(2*(u2+1)^2*(sqrt((u2+1)^2+(1-u1)^2)-sqrt(2)))/((u2+1)^2+(1-u1)^2)^(3/2)
    +(2*(u2+1)^2)/((u2+1)^2+(1-u1)^2)
    p4: Sec-Derivative u1, u1:
    (%o75) "p4: Sec-Derivative u1, u1:"
    (%o76) (2*(sqrt((u3-u4)^2+(-u2+u1+1)^2)-1))/sqrt((u3-u4)^2+(-u2+u1+1)^2)
    +(2*(-u2+u1+1)^2)/((u3-u4)^2+(-u2+u1+1)^2)
    -(2*(-u2+u1+1)^2*(sqrt((u3-u4)^2+(-u2+u1+1)^2)-1))
    /((u3-u4)^2+(-u2+u1+1)^2)^(3/2)
    p4: Sec-Derivative u1, u2:
    (%o77) "p4: Sec-Derivative u1, u2:"
    (%o78) -((2*(sqrt((u3-u4)^2+(-u2+u1+1)^2)-1))/sqrt((u3-u4)^2+(-u2+u1+1)^2))
    -(2*(-u2+u1+1)^2)/((u3-u4)^2+(-u2+u1+1)^2)
    +(2*(-u2+u1+1)^2*(sqrt((u3-u4)^2+(-u2+u1+1)^2)-1))
    /((u3-u4)^2+(-u2+u1+1)^2)^(3/2)
    p4: Sec-Derivative u2, u1:
    (%o79) "p4: Sec-Derivative u2, u1:"
    (%o80) -((2*(sqrt((u3-u4)^2+(-u2+u1+1)^2)-1))/sqrt((u3-u4)^2+(-u2+u1+1)^2))
    -(2*(-u2+u1+1)^2)/((u3-u4)^2+(-u2+u1+1)^2)
    +(2*(-u2+u1+1)^2*(sqrt((u3-u4)^2+(-u2+u1+1)^2)-1))
    /((u3-u4)^2+(-u2+u1+1)^2)^(3/2)
    p4: Sec-Derivative u2, u2:
    (%o81) "p4: Sec-Derivative u2, u2:"
    (%o82) (2*(sqrt((u3-u4)^2+(-u2+u1+1)^2)-1))/sqrt((u3-u4)^2+(-u2+u1+1)^2)
    +(2*(-u2+u1+1)^2)/((u3-u4)^2+(-u2+u1+1)^2)
    -(2*(-u2+u1+1)^2*(sqrt((u3-u4)^2+(-u2+u1+1)^2)-1))
    /((u3-u4)^2+(-u2+u1+1)^2)^(3/2)
    p5: Sec-Derivative u1, u1:
    (%o83) "p5: Sec-Derivative u1, u1:"
    (%o84) 0
    p5: Sec-Derivative u1, u2:
    (%o85) "p5: Sec-Derivative u1, u2:"
    (%o86) 0
    p5: Sec-Derivative u2, u1:
    (%o87) "p5: Sec-Derivative u2, u1:"
    (%o88) 0
    p5: Sec-Derivative u2, u2:
    (%o89) "p5: Sec-Derivative u2, u2:"
    (%o90) 0
    p6: Sec-Derivative u1, u1:
    (%o91) "p6: Sec-Derivative u1, u1:"
    (%o92) (2*(sqrt((-u4+u3+1)^2+(-u2+u1+1)^2)-sqrt(2)))/sqrt(
                                     (-u4+u3+1)^2
                                      +(-u2+u1+1)^2)
    +(2*(-u2+u1+1)^2)/((-u4+u3+1)^2+(-u2+u1+1)^2)
    -(2*(-u2+u1+1)^2*(sqrt((-u4+u3+1)^2+(-u2+u1+1)^2)-sqrt(2)))
    /((-u4+u3+1)^2+(-u2+u1+1)^2)^(3/2)
    p6: Sec-Derivative u1, u2:
    (%o93) "p6: Sec-Derivative u1, u2:"
    (%o94) -((2*(sqrt((-u4+u3+1)^2+(-u2+u1+1)^2)-sqrt(2)))
    /sqrt((-u4+u3+1)^2+(-u2+u1+1)^2))
    -(2*(-u2+u1+1)^2)/((-u4+u3+1)^2+(-u2+u1+1)^2)
    +(2*(-u2+u1+1)^2*(sqrt((-u4+u3+1)^2+(-u2+u1+1)^2)-sqrt(2)))
    /((-u4+u3+1)^2+(-u2+u1+1)^2)^(3/2)
    p6: Sec-Derivative u2, u1:
    (%o95) "p6: Sec-Derivative u2, u1:"
    (%o96) -((2*(sqrt((-u4+u3+1)^2+(-u2+u1+1)^2)-sqrt(2)))
    /sqrt((-u4+u3+1)^2+(-u2+u1+1)^2))
    -(2*(-u2+u1+1)^2)/((-u4+u3+1)^2+(-u2+u1+1)^2)
    +(2*(-u2+u1+1)^2*(sqrt((-u4+u3+1)^2+(-u2+u1+1)^2)-sqrt(2)))
    /((-u4+u3+1)^2+(-u2+u1+1)^2)^(3/2)
    p6: Sec-Derivative u2, u2:
    (%o97) "p6: Sec-Derivative u2, u2:"
    (%o98) (2*(sqrt((-u4+u3+1)^2+(-u2+u1+1)^2)-sqrt(2)))/sqrt(
                                     (-u4+u3+1)^2
                                      +(-u2+u1+1)^2)
    +(2*(-u2+u1+1)^2)/((-u4+u3+1)^2+(-u2+u1+1)^2)
    -(2*(-u2+u1+1)^2*(sqrt((-u4+u3+1)^2+(-u2+u1+1)^2)-sqrt(2)))
    /((-u4+u3+1)^2+(-u2+u1+1)^2)^(3/2)
    p7: Sec-Derivative u1, u1:
    (%o99) "p7: Sec-Derivative u1, u1:"
    (%o100) (2*(sqrt((-u4+u3+1)^2+(u2-u1+1)^2)-sqrt(2)))/sqrt(
                                     (-u4+u3+1)^2+(u2-u1+1)^2)
    +(2*(u2-u1+1)^2)/((-u4+u3+1)^2+(u2-u1+1)^2)
    -(2*(u2-u1+1)^2*(sqrt((-u4+u3+1)^2+(u2-u1+1)^2)-sqrt(2)))
    /((-u4+u3+1)^2+(u2-u1+1)^2)^(3/2)
    p7: Sec-Derivative u1, u2:
    (%o101) "p7: Sec-Derivative u1, u2:"
    (%o102) -((2*(sqrt((-u4+u3+1)^2+(u2-u1+1)^2)-sqrt(2)))
    /sqrt((-u4+u3+1)^2+(u2-u1+1)^2))
    -(2*(u2-u1+1)^2)/((-u4+u3+1)^2+(u2-u1+1)^2)
    +(2*(u2-u1+1)^2*(sqrt((-u4+u3+1)^2+(u2-u1+1)^2)-sqrt(2)))
    /((-u4+u3+1)^2+(u2-u1+1)^2)^(3/2)
    p7: Sec-Derivative u2, u1:
    (%o103) "p7: Sec-Derivative u2, u1:"
    (%o104) -((2*(sqrt((-u4+u3+1)^2+(u2-u1+1)^2)-sqrt(2)))
    /sqrt((-u4+u3+1)^2+(u2-u1+1)^2))
    -(2*(u2-u1+1)^2)/((-u4+u3+1)^2+(u2-u1+1)^2)
    +(2*(u2-u1+1)^2*(sqrt((-u4+u3+1)^2+(u2-u1+1)^2)-sqrt(2)))
    /((-u4+u3+1)^2+(u2-u1+1)^2)^(3/2)
    p7: Sec-Derivative u2, u2:
    (%o105) "p7: Sec-Derivative u2, u2:"
    (%o106) (2*(sqrt((-u4+u3+1)^2+(u2-u1+1)^2)-sqrt(2)))/sqrt(
                                     (-u4+u3+1)^2+(u2-u1+1)^2)
    +(2*(u2-u1+1)^2)/((-u4+u3+1)^2+(u2-u1+1)^2)
    -(2*(u2-u1+1)^2*(sqrt((-u4+u3+1)^2+(u2-u1+1)^2)-sqrt(2)))
    /((-u4+u3+1)^2+(u2-u1+1)^2)^(3/2)
