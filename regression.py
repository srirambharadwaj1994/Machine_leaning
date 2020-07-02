import csv
import random
import math
import operator
import numpy as np


def euclideanDistance(instance1, instance2):
    distance = 0
    distance = np.sqrt(sum((instance1 - instance2) ** 2))
    return distance


def manhattandistance(instance1, instance2):
    return sum(abs(instance1 - instance2))


def minkowskidistance(instance1, instance2 , p=2):
    return sum( abs(instance1 - instance2) ** p) ** (1 / p)


def hammingdistance(instance1, instance2):
    assert len(instance1) == len(instance2)
    return sum(instance1 != instance2 for instance1, instance2 in zip(instance1, instance2))


def getNeighbors(cancer_training, testInstance, k):
    distances = []
    # length = len(testInstance) - 1
    # print(length)
    for x in range(len(cancer_training)):
        dist = manhattandistance(testInstance, cancer_training[x])
        distances.append((cancer_training[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    k= 3
    summing = 0
    average_mean = []
    for x in range(len(neighbors)):
        summing = np.sum(neighbors)
    average_mean = summing / k

    return average_mean


def main():
    # prepare data
    cancer_training = []
    cancer_test = []

    # loadDataset('iris.data', split, trainingSet, testSet)
    cancer_training = np.genfromtxt("regressionData/trainingData.csv", delimiter=',')
    cancer_test = np.genfromtxt("regressionData/testData.csv", delimiter=',')
    print('Train set: ' + repr(len(cancer_training)))
    print('Test set: ' + repr(len(cancer_test)))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(cancer_test)):
        neighbors = getNeighbors(cancer_training, cancer_test[x], k)
        result = getResponse(neighbors)
        predictions.append(np.array(result))
        print('> predicted=' + repr(result) + ', actual=' + repr(cancer_test[x][-1]))
    sum_square = np.sum(sum((result - cancer_test) ** 2))
    # print(sum_square)
    total_sum = np.sum((sum_square - cancer_test) ** 2)
    # print(total_sum)
    r = 1 - (sum_square/total_sum)
    # print(r)
    accuracy = r
    print('Accuracy: ' + repr(accuracy) + '%')


main()
