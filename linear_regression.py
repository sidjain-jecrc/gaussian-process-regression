import scipy.io as sp
import scipy.spatial as ssp
import operator
import random

coeff = []
learningRate = 0.01


def init_coeff():
    for i in range(301):
        # coeff[i] = random.random()
        coeff.append(1.0)


def predict(x):
    summation = 0.0
    for i in range(len(x)):
        summation += x[i] * coeff[i + 1]
    # print(summation+coeff[0])
    return summation + coeff[0]


def linearRegressionSGD(trainingData, classLabel):
    sizeData = len(trainingData)
    print("In linearRegressionSGD")
    print(sizeData)
    for i in range(2):
        error = predict(trainingData[i]) - classLabel[i]
        # print(error)
        for ite in range(len(coeff)):
            coeff[ite] = coeff[ite] - learningRate * error * trainingData[i][ite - 1]
        coeff[0] = coeff[0] - learningRate * error


def rmse(validationData, validationOutput):
    length = len(validationData)
    sumerror = 0.0
    error = 0.0
    for i in range(length):
        subject = validationData[i]
        error = predict(subject) - validationOutput[i]
        sumerror += error * error
    sumerror = sumerror / length
    print(sumerror ** (1 / 2))


if __name__ == '__main__':
    readinput = sp.loadmat('data.mat')
    mapInputTrain = {}
    mapTargetTrain = {}
    for i in range(840):
        mapInputTrain[i] = readinput['input_train'][i]
        mapTargetTrain[i] = readinput['target_train'][i][0]
    mapInputVal = {}
    mapTargetVal = {}
    for i in range(1120):
        mapInputVal[i] = readinput['input_val'][i]
        mapTargetVal[i] = readinput['target_val'][i][0]
    mapInputTest = {}
    mapTargetTest = {}
    for i in range(560):
        mapInputTest[i] = readinput['input_test'][i]

    init_coeff()
    # print(coeff)
    linearRegressionSGD(mapInputVal, mapTargetVal)
    # print(coeff)
    rmse(mapInputTrain, mapTargetTrain)
