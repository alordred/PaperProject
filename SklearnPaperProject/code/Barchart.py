# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import crossValidation

def bar(values, Offsets):
    means_men = tuple(values)
    std_men = tuple(Offsets)
    names = []
    for clf, clfname ,color in crossValidation.algorithms:
        names.append(str(clfname))
    n_groups = len(means_men)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = ax.bar(index, means_men, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_men, error_kw=error_config,
                    label='Men')
    ax.set_xlabel('Group')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(tuple(names))
    ax.legend()
    fig.tight_layout()
    plt.show()
