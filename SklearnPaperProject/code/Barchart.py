# coding=utf-8
from timer import Timer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import crossValidation
from sklearn.ensemble import RandomForestClassifier
import math
import plottool
import config

def barImportance(X, y, max_k ,featureNames):
    forest = RandomForestClassifier(n_estimators=max_k)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    # for f in range(X.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    newFeatureNames = featureNames[0:48]
    plt.bar(range(len(newFeatureNames)), importances[indices],
            color="r", yerr=np.var(std[indices]/2.0), align="center")
    plt.xticks(range(len(newFeatureNames)), indices)
    # plt.xlim([-1, featureNames[1]])
    plottool.plot_saveEsp(config.featureImportance_dir)

def barCompare(values, Offsets):
    means_men = tuple(values)
    std_men = tuple(Offsets)
    names = []
    for clf, clfname ,color in crossValidation.algorithms:
        names.append(str(clfname))
    n_groups = len(means_men)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.6
    ax.bar(range(len(means_men)), means_men, yerr=std_men, edgecolor='black', hatch='/')
    ax.set_xlabel('Group')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(tuple(names))
    ax.legend()
    fig.tight_layout()
    plottool.plot_saveEsp(config.algorithmCompare_dir)
