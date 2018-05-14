# coding=utf-8

from sklearn.ensemble import RandomForestClassifier as RF, GradientBoostingRegressor as GBDT
from sklearn.ensemble import ExtraTreesClassifier as ETDT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import classifiers

from sklearn.tree import DecisionTreeClassifier



k_range = range(5,60)

def crossValidateMain(xs, ys):
    for clf, clfname ,color in algorithms:
        crossValidate(xs, ys, clf, clfname, color)
    draw()

def crossValidate(xs, ys, clf, clfname, color):
    k_scores = []
    max_score = 0
    print clfname
    for k in k_range:
        print "process:" + str(k)
        scores = cross_val_score(clf(k), xs, ys, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
        if scores.mean() > max_score:
            max_score = scores.mean()
            max_k = k
    print clfname + "max score : " + str(max_score)
    print "max k : " + str(max_k)
    plt.plot(k_range, k_scores, 'o-', color=color, label=clfname)
    return

def draw():
    plt.xlabel('Value of K for RandomForest')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

def new_rf(k):
    args = {
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

def new_gbdt(k):
    args = {"n_estimators": 400,
            "max_depth": k,
            "max_features": "sqrt",
            }
    return GBDT(**args)

def new_ab(k):
    args = {"n_estimators": k,
            }
    return AB(**args)

algorithms = [(new_rf, "RandomForestClassifier", "r"),
              (new_etdt, "ExtraTreesClassifier", "g"),
              (new_knn, "KNN", "b"),
              (new_etdt, "ETDT", "c"),
              (new_sgd, "SGD", "k"),
              (new_ab, "AdaBoostClassifier", "m"),
              # (new_gbdt, "GBDT", "m"),
              ]