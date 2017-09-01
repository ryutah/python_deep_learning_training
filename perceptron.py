import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([.5, .5])
    b = -.7
    tmp = np.sum(x*w) + b
    return 0 if tmp <= 0 else 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-.5, -.5])
    b = .7
    tmp = np.sum(x*w) + b
    return 0 if tmp <= 0 else 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([.5, .5])
    b = -.2
    tmp = np.sum(w*x) + b
    return 0 if tmp <= 0 else 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
