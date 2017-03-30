from math import exp
import scipy.io as sp
import scipy.spatial as ssp
import operator
import random
import numpy as np
from numpy.linalg import inv

sigmaN = 0.0464386
sigmaF = 0.08672187
distL = 0.2598845


def cov(x1, x2):
    sum = 0.0
    for each in range(len(x1)):
        sum += pow((x1[each] - x2[each]), 2)
    sum = -1 * sum
    sum = sum / (2 * distL * distL)
    Exp = exp(sum) * sigmaF * sigmaF
    if x1.all() == x2.all():
        Exp += sigmaN * sigmaN
    return Exp


def formK(InputArray):
    length = len(InputArray)
    # print(length)
    Kmatrix = np.ones([length, length])

    for iter0 in range(length):
        for iter1 in range(length):
            # print(iter0," ",iter1)
            Kmatrix[iter0, iter1] = cov(InputArray[iter0], InputArray[iter1])

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


if __name__ == '__main__':
    readinput = sp.loadmat('data.mat')
    input_train = readinput['input_train']
    target_train = readinput['target_train']
    input_val = readinput['input_val']
    target_val = readinput['target_val']
    input_test = readinput['input_test']

    result = predict(input_train, target_train, input_val)
    print(result)

    rmse(result, target_val)
