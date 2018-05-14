# coding=utf-8

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import classifiers
k_range = range(1,10)
k_scores = []

def crossValidateMain(xs, ys):
    for clf, clfname in classifiers.clfs:
        crossValidate(xs, ys, clf, clfname)

def crossValidate(xs, ys, clf, clfname):
    max_score = 0
    print clfname
    for k in k_range:
        print "process:" + str(k)
        rf = RandomForestClassifier(n_estimators=k)
        scores = cross_val_score(clf, xs, ys, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
        if scores.mean > max_score:
            max_score = scores.mean()
    print "max score: " + str(max_score)
    plt.plot(k_range, k_scores, 'o-', color="g", label="RandomForestClassifier")
    draw()
    return

def draw():
    plt.xlabel('Value of K for RandomForest')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()