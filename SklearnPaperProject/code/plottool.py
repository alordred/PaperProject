#! encoding=utf-8

# Creation Date: 2018-04-21 01:39:09
# Created By: Heyi Tang

from matplotlib import pyplot as plt
import os

def plot_seqs(seqs, fname = None):
    for x, y, label in seqs:
        if not isinstance(x, list):
            x = list(x)
        if not isinstance(y, list):
            y = list(y)
        plt.plot(x, y, label = label, marker = "o")
    plt.legend(loc = "best")
    if fname is not None:
        figdir = "/".join(fname.split("/")[:-1])
        if figdir != "" and not os.path.exists(figdir):
            os.makedirs(figdir)
        plt.savefig(fname)
    plt.close("all")

def plot_saveEsp(dir):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(dir)
