"""#The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python
#NumPy’s main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of the
same type, indexed by a tuple of positive integers"""

import operator
import numpy as np


def euclideanDistance(instance1, instance2):
    """#inistializing the variable distance
    # assigning the euclidian formula to distance and return the value ( using the np.sqrt method to calculate the
    square root)"""
    distance = 0
    distance = np.sqrt(sum((instance1 - instance2) ** 2))
    return distance


def getNeighbors(cancer_training, testInstance, k):
    """#initializing distances as list
    #calculating the distance between the neighbours using the euclideanDistance
    #and appending the values of the data(neighbous and distance(dist) into distances[]
    #sorting the distances list using sort method from operator module
    #appending the distances based on the sort to neighbours """
    distances = []
    for x in range(len(cancer_training)):
        dist = euclideanDistance(testInstance, cancer_training[x])
        distances.append((cancer_training[x], dist))
    # Given a list lst of tuples, you can sort by the ith element (using module operator)
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    """#taking the votes count of the neighbours which is inherited from getneighbour function and classifying"""
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(cancer_test, predictions):
    """#getting the accuracy count based on the prediction value which is calcualted inside the main function"""
    correct = 0
    for x in range(len(cancer_test)):
        if cancer_test[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(cancer_test))) * 100.0


def main():
    """#declaration of two lists to hold test and train data
    #loading data set into the declared data set(cancer_training,cancer_test) using np.genfromtxt as a list
    #printing the count of the dataset values
    #assigning the value of k(to pick neighbours)
    #passing on the values to neighbours from getneighbour funtion
    #and their value to result that will be considered as prediction
    #comparision of predicted and actual values
    #generating accuracy based in actual an predicted values"""

    cancer_training = []
    cancer_test = []

    # loadDataset('iris.data', split, trainingSet, testSet)
    cancer_training = np.genfromtxt("cancer2/trainingData2.csv", delimiter=',')
    cancer_test = np.genfromtxt("cancer2/testData2.csv", delimiter=',')
    print('Train set: ' + repr(len(cancer_training)))
    print('Test set: ' + repr(len(cancer_test)))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(cancer_test)):
        neighbors = getNeighbors(cancer_training, cancer_test[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(cancer_test[x][-1]))
    accuracy = getAccuracy(cancer_test, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()
