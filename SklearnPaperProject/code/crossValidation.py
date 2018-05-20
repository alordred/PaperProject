# coding=utf-8

from sklearn.ensemble import RandomForestClassifier as RF, GradientBoostingClassifier as GBDT
from sklearn.ensemble import ExtraTreesClassifier as ETDT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.svm import LinearSVC as SVC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import xlwt
from datetime import datetime
import Barchart as barchart
import numpy as np
import classifiers

from sklearn.tree import DecisionTreeClassifier



k_range = range(100,101)
xls_dir = "/Users/lihongsheng/Desktop/MyProject/PaperProject/SklearnPaperProject/xls/"

workbook = xlwt.Workbook()
sheet = workbook.add_sheet("Data")
j = 0
charColumnValue = []
charColumnOffset = []
importances = []
indices = []
def EvaluationFeatureImportance(xs, ys, max_k):
    global importances, indices
    forest = RF(n_estimators=max_k, random_state=0, n_jobs=-1)
    forest.fit(xs, ys)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(len(importances)):
        print(importances[indices[f]])
    barchart.barImportance(importances, indices, forest)

def crossValidateMain(xs, ys):
    global charColumnValue, charColumnOffset
    global importances, indices
    for clf, clfname ,color in algorithms:
        print "for"
        global j
        crossValidate(xs, ys, clf, clfname, color)
        j = j + 1
    # draw()
    barchart.barCompare(charColumnValue, charColumnOffset)

def crossValidate(xs, ys, clf, clfname, color):
    global charColumnValue, charColumnOffset
    row0 = 0
    row1 = 1
    k_scores = []
    maxAverageScore = 0
    maxOffest = 0
    max_k = 0
    print clfname
    for k in k_range:
        print "process:" + str(k)
        scores = cross_val_score(clf(k), xs, ys, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
        if scores.mean() > maxAverageScore:
            max_k = k
            maxAverageScore = scores.mean()
            maxOffest = max(scores) - min(scores)
            w = 1
    if clfname == "RandomForest":
        EvaluationFeatureImportance(xs, ys, max_k)
    print clfname + "maxAverageScore score : " + str(maxAverageScore)
    print "max k : " + str(max_k)
    print "maxOffest : " + str(maxOffest)
    charColumnValue.append(maxAverageScore)
    charColumnOffset.append(maxOffest/2.0)
    sheet.write(row0, j, clfname)
    sheet.write(row1, j, str(maxAverageScore))
    # plt.plot(k_range, k_scores, 'o-', color=color, label=clfname)
    return

# 根据不同的参数画图
def draw():
    xls_file = xls_dir + "%s_data_%s.xls" % ("data", datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    workbook.save(xls_file)
    plt.xlabel('Value of K for RandomForest')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

def new_rf(k):
    args = {
            "max_depth": 100,
            "n_estimators": k,
            }
    return RF(**args)

def new_etdt(k):
    args = {"n_estimators": k,
            }
    return ETDT(**args)

def new_knn(k):
    args = {"n_neighbors": k,
            }
    return KNN(**args)

def new_etdt(k):
    args = {"n_estimators": k,
            }
    return ETDT(**args)

def new_sgd(k):
    args = {"n_iter": k,
            }
    return SGD(**args)

def new_ab(k):
    args = {"n_estimators": k,
            }
    return AB(**args)

def new_svc(k):
    args = {
            }
    return SVC(**args)

def new_gnb(k):
    args = {
            }
    return GNB(**args)

def new_gbdt(k):
    args = {"n_estimators": k,
            }
    return GBDT(**args)

algorithms = [(new_rf, "RandomForest", "r"),
              (new_etdt, "ExtraTrees", "g"),
              (new_knn, "KNN", "b"),
              (new_etdt, "ETDT", "c"),
              (new_sgd, "SGD", "k"),
              (new_ab, "AdaBoost", "m"),
              (new_svc, "SVC", "y"),
              (new_gnb, "GaussianNB", "w"),
              (new_gbdt, "GradientBoost", "#DB7093")
              ]