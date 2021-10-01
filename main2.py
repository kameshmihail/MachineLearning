import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rd
from scipy.optimize import fmin_tnc

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=1000):
        self.max_depth = max_depth


    def fit(self, X, y, rand=0):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, rand)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def random_best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        l = list(range(self.n_features_))
        random.shuffle(l)
        l = np.array(l)
        feat_num = int(np.ceil(np.sqrt(self.n_features_)))
        l = np.resize(l, feat_num)

        for idx in l:
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0, rand=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            if rand:
                idx, thr = self.random_best_split(X, y)    
            else:
                idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

class RandomTreeClassifier:
    def __init__(self, tree_num=10):
        self.tree_num = tree_num
        self.trees = []

    def get_sub_x(self, X,y):
        x_size = len(X)
        sample = []
        new_y = []
        while len(sample) < x_size:
            ind = rd.randrange(x_size)
            sample.append(X[ind])
            new_y.append(y[ind])
        sample = np.array(sample)
        new_y = np.array(new_y)
        return sample, new_y

    def fit(self, X, y):
        for i in range(self.tree_num):
            tree = DecisionTreeClassifier()
            X1, Y1 = self.get_sub_x(X, y)
            tree.fit(X1,Y1, 1)
            self.trees.append(tree)

    def predict(self, X):
        res = np.zeros(len(X))
        for i in range(self.tree_num):
            res += self.trees[i].predict(X)

        for i in range(len(res)):
            if (res[i] / self.tree_num < 0.5):
                res[i] = 0
            else:
                res[i] = 1
        return res

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    return np.dot(x, theta)

def probability(theta, x):
    return sigmoid(net_input(theta, x))


def cost_function(theta, x, y):
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(theta, x, y):
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

def fit(x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta,
                  fprime=gradient,args=(x, y), messages=0)
    return opt_weights[0]

def prediction(x, param):
    theta = param[:, np.newaxis]
    return probability(theta, x)

def predict(res, probab_threshold=0.5):
    predicted_classes = []
    for pred in res:
        if pred >= probab_threshold:
            predicted_classes += [1]
        else:
            predicted_classes += [0]
    return predicted_classes

def accur(res, trueX):
	count = 0
	for i in range(len(trueX)):
		if ((res[i] >=0.5 and trueX[i] == 1) or
			(res[i] < 0.5 and trueX[i] == 0)):
			count+=1
	return count / len(trueX)
	
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

    testX = []
    testY = []
    trainX = []
    trainY = []

    for i in range(len(trainData)):
        trainX.append(trainData[i][0])
        trainY.append(trainData[i][1])

    for i in range(len(testData)):
        testX.append(testData[i][0])
        testY.append(testData[i][1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)

    return trainX, trainY, testX, testY

if __name__ == '__main__':
    dataset = pd.read_csv('../alc2.csv')
    trainX, trainY, testX, testY = dataPreparing(dataset, 0.85)
    model = LogisticRegression(solver='liblinear', random_state=0).fit(trainX, trainY)
    print("1.1: SKlearn Logistic Regression\n")
    print('SKlearn log accuracy: ',model.score(testX,testY))
    pred = model.predict(testX)
    tn, fp, fn, tp = confusion_matrix(testY, pred).ravel()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("SKlearn log precision = ", pre)
    print("SKlearn log recall = ", rec)


    trainLog = np.c_[np.ones(len(trainX)), trainX]
    testLog = np.c_[np.ones(len(testX)), testX]
    theta = np.zeros((trainLog.shape[1], 1))
    param = fit(trainLog, trainY, theta)
    res = prediction(testLog, param)
    print("\n1.2: Custom Logistic Regression\n")
    print('Custom log accuracy: ', accur(res, testY))
    pred = predict(res)
    tn, fp, fn, tp = confusion_matrix(testY, pred).ravel()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("Custom log precision = ", pre)
    print("Custom log recall = ", rec)
    
    
    res = prediction(trainLog, param)
    pred_class = predict(res)
    print('Custom log train accuracy:', accur(pred_class, trainY))

    print("\n2.1: SKlearn DecisionTreeClassifier\n")
    model=tree.DecisionTreeClassifier()
    model.fit(trainX,trainY)
    print('SKlearn Dtree accuracy:', model.score(testX,testY))
    tn, fp, fn, tp = confusion_matrix(testY, pred).ravel()
    pred = model.predict(testX)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("SKlearn Dtree precision = ", pre)
    print("SKlearn Dtree recall = ", rec)
    
    
    print("\n2.2: Custom Decision Tree Classifier\n")


    model = DecisionTreeClassifier(max_depth=5)
    model.fit(trainX, trainY)
    print('Custom Dtree accuracy:', accur(pred, testY))
    pred = model.predict(testX)
    tn, fp, fn, tp = confusion_matrix(testY, pred).ravel()
    pred = model.predict(testX)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("Custom Dtree precision = ", pre)
    print("Custom Dtree recall = ", rec)
    
    
    pred_class = model.predict(trainX)
    print('Custom Dtree train accuracy:', accur(pred_class, trainY))



    print("\n3.1: SKlearn Random Forest Classifier\n")
    model = RandomForestClassifier(n_estimators=10)
    model.fit(trainX, trainY)

    print('SKlearn RF accuracy:', model.score(testX,testY))
    pred = model.predict(testX)
    tn, fp, fn, tp = confusion_matrix(testY, pred).ravel()
    pred = model.predict(testX)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("SKlearn RF precision = ", pre)
    print("SKlearn RF recall = ", rec)
    
    

    print("\n3.2: Custom Random Forest Classifier\n")
    model = RandomTreeClassifier(5)
    model.fit(trainX, trainY)

    print('Custom RF accuracy: ', accur(pred, testY))
    pred = model.predict(testX)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("Custom RF precision = ", pre)
    print("Custom RF recall = ", rec)
    

    pred_class = model.predict(trainX)
    print('Custom RF train accuracy:', accur(pred_class, trainY))
