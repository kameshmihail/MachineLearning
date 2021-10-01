from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import math
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def dist (a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def myKNN (testData, trainData, k):
    result = []
    for testPoint in testData:
        distance = [[dist(testPoint[0], trainData[i][0]), trainData[i][1]] for i in range(len(trainData))]
        q = [0 for i in range(numberOfClasses)]

        for closedPoint in sorted(distance)[0:k]:
            q[closedPoint[1]] += 1 / closedPoint[0]**2

        result.append(q.index(max(q)))
    return result

def mean(numbers):
    return sum(numbers) / float(len(numbers))  

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = []
    for i in range(len(dataset[0][0])):
        feat = [dataset[j][0][i] for j in range(len(dataset))]
        summaries.append((mean(feat), stdev(feat)))
    return summaries

def summarizeByClass(dataset):
    summaries = {}
    separated = {}
    for inst in dataset:
        if (inst[-1] not in separated):
            separated[inst[-1]] = []
        separated[inst[-1]].append(inst)

    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-((x-mean)**2/(2*stdev**2)))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for instance in testSet:
        result = predict(summaries, instance[0])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))

def myBayes(testData, trainData):
    summaries = summarizeByClass(trainData)
    predictions = getPredictions(summaries, testData)
    return predictions

def dataPreparing(dataset, split):
    X1 = dataset[10:110]['MMSE'].values
    Y1 = dataset[10:110]['eTIV'].values
    X2 = dataset[200:300]['MMSE'].values
    Y2 = dataset[200:300]['eTIV'].values

    req_data = []
    for i in range(len(X1)):
        req_data.append([[X1[i], Y1[i]],0])

    for i in range(len(X2)):
        req_data.append([[X2[i], Y2[i]],1])

    trainData =[]
    testData = []
    for i in range (len(req_data)):
        if rd.random() < split:
            trainData.append(req_data[i])
        else:
            testData.append(req_data[i])
    
    return trainData, testData

if __name__ == '__main__':
    dataset = pd.read_csv('../alc2.csv')
    numberOfClasses = 2
    k = 5
    trainData, testData = dataPreparing(dataset, 0.85)

    result = myKNN(testData, trainData, k)
    predictions = myBayes(testData, trainData)

    accuracy = getAccuracy(testData, predictions)
    accuracyK = getAccuracy(testData, result)
    accuracyB = getAccuracy(testData, predictions)
    
    print("1.1 Custom KNeighborsClassifier\n")
    print("Custom KNN accuracy = ", accuracyK)

    tn, fp, fn, tp = confusion_matrix([testData[i][1] for i in range(len(testData))], result).ravel()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("Custom KNN precision = ", pre)
    print("Custom KNN recall = ", rec)
    
    print("\n1.2 SKlearn KNN\n")
    
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    X = []
    Y = []
    for i in range(len(trainData)):
        X.append(trainData[i][0])
        Y.append(trainData[i][1])

    knn.fit(X, Y)
    answ1 = knn.predict([testData[i][0] for i in range(len(testData))])
    print('SKlearn KNN accuracy = ', getAccuracy(testData, answ1))
    tn, fp, fn, tp = confusion_matrix([testData[i][1] for i in range(len(testData))], answ1).ravel()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("SKlearn KNN precision = ", pre)
    print("SKlearn KNN recall = ", rec)
    
    print("\n2.1 Custom Naive Bayes classifier\n")
    print("Custom NB accuracy = ", accuracyB)

    tn, fp, fn, tp = confusion_matrix([testData[i][1] for i in range(len(testData))], predictions).ravel()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("Custom NB precision = ", pre)
    print("Custom NB recall = ", rec)
    
    print("\n2.2 SKlearn Naive Bayes classifier\n")
    nb = GaussianNB()
    nb.fit(X, Y)
    answ2 = nb.predict([testData[i][0] for i in range(len(testData))])
    print('SKlearn NB accuracy = ', getAccuracy(testData, answ2))
    
    tn, fp, fn, tp = confusion_matrix([testData[i][1] for i in range(len(testData))], answ2).ravel()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("SKlearn NB precision = ", pre)
    print("SKlearn NB recall = ", rec)
    
