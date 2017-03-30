from math import exp
import scipy.io as sp
import scipy.spatial as ssp
import operator
import random
import numpy as np
from numpy.linalg import inv
import math
import scipy.optimize as op


def cov(x1, x2, sigmaF, sigmaN, distL):
    sum = 0.0
    for each in range(len(x1)):
        sum += pow((x1[each] - x2[each]), 2)
    sum = -1 * sum
    sum = sum / (2 * distL * distL)
    Exp = exp(sum) * sigmaF * sigmaF
    if x1.all() == x2.all():
        Exp += sigmaN * sigmaN
    return Exp


def formK(InputArray, sigmaF, sigmaN, distL):
    length = len(InputArray)
    # print(length)
    Kmatrix = np.ones([length, length])

    for iter0 in range(length):
        for iter1 in range(length):
            # print(iter0," ",iter1)
            Kmatrix[iter0, iter1] = cov(InputArray[iter0], InputArray[iter1], sigmaF, sigmaN, distL)

    return Kmatrix


def formKStar(InputArray, Xstar):
    length = len(InputArray)
    Kstar = np.ones([length, 1])
    for iter0 in range(length):
        Kstar[iter0] = cov(InputArray[iter0], Xstar)
    return Kstar


def rmse(validationData, validationOutput):
    length = len(validationData)
    sumerror = 0.0
    error = 0.0
    for i in range(length):
        error = validationData[i] - validationOutput[i]
        sumerror += error * error
    sumerror = sumerror / length
    print(sumerror ** (1 / 2))


def TrainFunction(input_train, target_train):
    Kmatrix = formK(input_train)
    invKmatrix = inv(Kmatrix)
    y = readinput['target_train']
    return np.matmul(invKmatrix, y)


def predict(input_train, target_train, input_val):
    length = len(input_val)
    secondterm = TrainFunction(input_train, target_train)
    result = np.ones([length, 1])
    for iter0 in range(length):
        Kstar = formKStar(input_train, input_val[iter0])
        transposeKstar = np.transpose(Kstar)
        result[iter0] = np.matmul(transposeKstar, secondterm)[0]
    return result


# reading input .mat file to retrieve various input matrices
readinput = sp.loadmat('data.mat')
input_train = readinput['input_train']
target_train = readinput['target_train']
input_val = readinput['input_val']
target_val = readinput['target_val']
input_test = readinput['input_test']

globaly = target_train
globaln = len(input_train)
count = 0

thetalist = [sigmaF, sigmaN, distL]
ytranspose = np.transpose(globaly)

def costfunc(thetalist):
    print(thetalist)
    kmatrix = formK(input_train, thetalist[0], thetalist[1], thetalist[2])
    invmatrix = inv(kmatrix)
    print("-------------------------------------------------------------")
    print("inverse matrix")
    print(invmatrix.shape)
    r = np.matmul(ytranspose, invmatrix)
    print("==============================================================")
    print("R matrix")

    print(r.shape)
    print(r)
    term1 = np.matmul(r, globaly)
    print("-------------------------------------------------------------")
    print("term1")

    print(term1)

    (sign, logdet) = np.linalg.slogdet(kmatrix)
    print("sign", sign)
    print("logdet", logdet)
    term2 = logdet
    print("-------------------------------------------------------------")
    print("term2")
    print(term2)
    # term2 = np.log(term2)

    term3 = globaln * np.log(2 * math.pi)
    result = (0.5) * (term1 + term2 + term3)
    print(result)
    return result


Result = op.minimize(fun=costfunc, x0=thetalist)
print('1')
print(Result.x)
print(Result.success)
print(Result.message)
