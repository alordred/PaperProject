#! encoding=utf-8

from sklearn.ensemble import RandomForestClassifier as RF, GradientBoostingRegressor as GBDT
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.svm import LinearSVC as SVC
from sklearn.svm import LinearSVR as SVR
from sklearn.naive_bayes import GaussianNB as GNB

def new_rf():
    args = {"max_depth":13,
            "random_state": 0,
            "n_estimators":49,
            "class_weight":"balanced_subsample",
           # "max_features": None,
            }
    return RF(**args)

def new_gbdt():
    args = {"n_estimators": 400,
            "max_depth": 10,
            "max_features": "sqrt",
            }
    return GBDT(**args)

def new_knn():
    args = {"n_neighbors": 5
            }
    return KNN(**args)

def new_dt():
    args = {
            }
    return DT(**args)

def new_svc():
    args = {
            }
    return SVC(**args)

# 有些算法跑起来会有预测分母为零
def new_ab():
    args = {
            }
    return AB(**args)


clfs = [(new_rf(), "RandomForest"),
        (new_gbdt(), "GBDT"),
        (new_knn(), "KNN"),
        (new_dt(), "DecisionTree"),
        (new_svc(), "SVC"),
        ]
