#! encoding=utf-8

from sklearn.ensemble import RandomForestClassifier as RF, GradientBoostingRegressor as GBDT
from sklearn.ensemble import AdaBoostClassifier as AB, ExtraTreesClassifier as ETDT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.svm import LinearSVC as SVC
from sklearn.svm import LinearSVR as SVR
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.linear_model import PassiveAggressiveClassifier as PA

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

def new_etdt():
    args = {
            }
    return ETDT(**args)

# 警告
def new_pa():
    args = {
            }
    return PA(**args)

# 下面都是改了pre的数值不为0
def new_mlp():
    args = {
            }
    return MLP(**args)

# 警告
def new_sgd():
    args = {
            }
    return SGD(**args)

# 数组越界,尚未修好
def new_svr():
    args = {
            }
    return SVR(**args)

def new_gnb():
    args = {
            }
    return GNB(**args)

# 有些算法跑起来会有预测分母为零
def new_ab():
    args = {"n_estimators": 400,
            }
    return AB(**args)


clfs = [(new_rf(), "RandomForest"),
        (new_gbdt(), "GBDT"),
        (new_knn(), "KNN"),
        (new_dt(), "DecisionTree"),
        (new_svc(), "SVC"),
        (new_ab(), "AdaBoost"),
        (new_mlp(), "MLP"),
        (new_sgd(), "SGD"),
        (new_gnb(), "GaussianNB"),
        (new_etdt(), "ExtraTrees"),
        (new_pa(), "PassiveAggressive"),
        (new_svr(), "SVR"),
        ]
