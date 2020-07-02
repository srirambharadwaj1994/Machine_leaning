import csv
import random
import math
import operator
import numpy as np
from collections import Counter


def euclideanDistance(instance1, instance2):
    distance = 0
    distance = np.sqrt(sum((instance1 - instance2) ** 2))
    return distance


def getNeighbors(cancer_training, testInstance, k):
    distances = []
    # length = len(testInstance) - 1
    # print(length)
    for x in range(len(cancer_training)):
        dist = euclideanDistance(testInstance, cancer_training[x])
        distances.append((cancer_training[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def vote_harmonic_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        class_counter[neighbors[index][2]] += 1/(index+1)
    labels, votes = zip(*class_counter.most_common())
    #print(labels, votes)
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)
# def getResponse(neighbors):
#     classVotes = {}
#     for x in range(len(neighbors)):
#         response = neighbors[x][-1]
#         if response in classVotes:
#             classVotes[response] += 1
#         else:
#             classVotes[response] = 1
#     sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
#     return sortedVotes[0][0]


def getAccuracy(cancer_test, predictions):
    correct = 0
    for x in range(len(cancer_test)):
        if np.array_equal(cancer_test[x][-1],predictions[x]):
            correct += 1
    return (correct / float(len(cancer_test))) * 100.0


def main():
    # prepare data
    cancer_training = []
    cancer_test = []

    # loadDataset('iris.data', split, trainingSet, testSet)
    cancer_training = np.genfromtxt("cancer2/trainingData2.csv", delimiter=',')
    cancer_test = np.genfromtxt("cancer2/testData2.csv", delimiter=',')
    print('Train set: ' + repr(len(cancer_training)))
    print('Test set: ' + repr(len(cancer_test)))
    # generate predictions
    predictions = []
    k = 25
    for x in range(len(cancer_test)):
        neighbors = getNeighbors(cancer_training, cancer_test[x], k)
        result = vote_harmonic_weights(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(cancer_test[x][-1]))
    accuracy = getAccuracy(cancer_test, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()
