import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold  # create indices for CV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import style

from plotify import Plotify

# read in the data
data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
data_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

# split input variables and labels
X_train = data_train[:, :-1]
y_train = data_train[:, -1]

X_test = data_test[:, :-1]
y_test = data_test[:, -1]

# Exercise 1

clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print('Exercise 1')

print('accuracy of 1-NN: ', accuracy)

# Exercise 2

cv = KFold(n_splits=5)
# loop over CV folds

# for train, test in cv.split(X_train):
#   X_trainCV, X_testCV, y_trainCV, y_testCV = X_train[train], X_train[test], y_train[train], y_train[test]
#   clf.fit(X_trainCV, y_trainCV)
#   accuracy = clf.score(X_testCV, y_testCV)
#   print(accuracy)

possible_ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
mean_scores = []

def find_best_k(X_train, X_test, y_train, y_test, possible_ks):
  for i, k in enumerate(possible_ks):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    scores = cross_val_score(clf, X_test, y_test, cv=5)
    # print('scores', scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    mean_score = scores.mean()
    mean_scores.append(mean_score)
  
  best_k = possible_ks[mean_scores.index(max(mean_scores))]
  return best_k

best_k = find_best_k(X_train, X_test, y_train, y_test, possible_ks)
print('(not normalized) best k is: ', best_k)

plotify = Plotify()

plotify.bar(
    x_list=possible_ks,
    y_list=mean_scores,
    xlabel='k',
    ylabel='Accuracy',
    title='Different Accuracies using different value for k (not normalized)',
    ymin=0.905,
    ymax=0.94,
    linewidth=1,
    block=False,
    use_x_list_as_xticks=True
)

# Exercise 3

clf = neighbors.KNeighborsClassifier(n_neighbors=best_k)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('(not normalized) best k accuracy: ', accuracy)

# Exercise 4


# version 1
scaler = preprocessing.StandardScaler().fit(X_train) 
X_trainN = scaler.transform(X_train)
X_testN = scaler.transform(X_test)
print('# Exercise 4')
print('version 1 is correct.')
print(scaler.mean_)

mean_scores = []

best_k = find_best_k(X_trainN, X_testN, y_train, y_test, possible_ks)

clf = neighbors.KNeighborsClassifier(n_neighbors=best_k)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('(normalized) Best k accuracy: ', accuracy)

plotify.bar(
    x_list=possible_ks,
    y_list=mean_scores,
    xlabel='value of k',
    ylabel='Accuracy',
    title='Different Accuracies using different value for k (normalized)',
    ymin=0.92,
    ymax=0.95,
    linewidth=1,
    block=False,
    use_x_list_as_xticks=True
)

plt.show()

