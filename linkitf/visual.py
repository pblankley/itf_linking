# Imports
import numpy as np
import scipy.interpolate
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import math
import healpy as hp
import collections
import astropy
from collections import defaultdict
from collections import Counter
import scipy.spatial
import pickle
from operator import add
import matplotlib.cm as cm
import matplotlib.colors as mlc


def number_clusters_plot(pix_runs,true_count,outpath=''):
    """ This function plots the number of clusters we find vs the
    cluster radius we define.  It is used as a visual for training.
    -------
    Args: pix_runs; the result of looping over various healpy pixel values
            in training and putting them in a dictionary.  To learn more
            about this data structure look at the docs for tuning.
          true_count; int, the true number of clusters in training.
            Get this value by summing up the values of true_count_dict from
            "accessible_clusters.
    -------
    Returns: None; plots the graph.
    """"
    for dt in np.arange(5, 30, 5):
        pixels=list(pix_runs.keys())
        ds = pix_runs[pixels[0]][dt][0]
        nclusters = pix_runs[pixels[0]][dt][1]
        nerrors = pix_runs[pixels[0]][dt][2]
        for pix in pixels[1:]:
            nclusters = list(map(add, nclusters, pix_runs[pix][dt][1]))
            nerrors = list(map(add, nerrors, pix_runs[pix][dt][2]))
        nclusters=np.array(nclusters)

        plt.plot(ds, nclusters, label=dt)

    plt.axhline(true_count, ls='dashed')
    plt.xlabel('d (cluster radius)')
    plt.ylabel('N clusters')
    plt.legend()
    plt.title('Number of clusters by cluster radius.')
    if outpath != '':
        plt.savefig(outpath)
    plt.show()


def number_errors_plot(pix_runs,outpath=''):
    """ This function plots the number of errors we have vs the
    cluster radius we define.  It is used as a visual for training.
    -------
    Args: pix_runs; the result of looping over various healpy pixel values
            in training and putting them in a dictionary.  To learn more
            about this data structure look at the docs for tuning.
    -------
    Returns: None; plots the graph.
    """"
    for dt in np.arange(5, 30, 5):
        pixels=list(pix_runs.keys())
        ds = pix_runs[pixels[0]][dt][0]
        nclusters = pix_runs[pixels[0]][dt][1]
        nerrors = pix_runs[pixels[0]][dt][2]
        for pix in pixels[1:]:
            nclusters = list(map(add, nclusters, pix_runs[pix][dt][1]))
            nerrors = list(map(add, nerrors, pix_runs[pix][dt][2]))
        nclusters=np.array(nclusters)
        nerrors=np.array(nerrors)

        plt.plot(ds, nerrors, label=dt)

    plt.xscale("log", nonposx='clip')
    plt.ylim((0,3000))
    plt.xlabel('d (cluster radius)')
    plt.ylabel('N errors')
    plt.text(0.0005, 1000, r'$\gamma=0.4$', fontsize=15)
    plt.legend()
    plt.title('Number of errors by cluster radius')
    if outpath != '':
        plt.savefig(outpath)
    plt.show()


def auc_plot(pix_runs,true_count,outpath=''):
    """ This function plots the number of errors we have as a percentage
    of the total true clusters vs the number of clusters we find as a
    percentage of the total true clusters. This approximates the AUC measure.
    It is used as a visual for training.
    -------
    Args: pix_runs; the result of looping over various healpy pixel values
            in training and putting them in a dictionary.  To learn more
            about this data structure look at the docs for tuning.
          true_count; int, the true number of clusters in training.
            Get this value by summing up the values of true_count_dict from
            "accessible_clusters.
    -------
    Returns: None; plots the graph.
    """"
    for dt in np.arange(5, 30, 5):
        pixels=list(pix_runs.keys())
        ds = pix_runs[pixels[0]][dt][0]
        nclusters = pix_runs[pixels[0]][dt][1]
        nerrors = pix_runs[pixels[0]][dt][2]
        for pix in pixels[1:]:
            nclusters = list(map(add, nclusters, pix_runs[pix][dt][1]))
            nerrors = list(map(add, nerrors, pix_runs[pix][dt][2]))
        nclusters=np.array(nclusters)
        nerrors=np.array(nerrors)

        plt.plot(nerrors/true_count, nclusters/true_count, label=dt)

    plt.xlim((0, 0.02))
    plt.ylim((0, 1))
    plt.xlabel('Error rate')
    plt.ylabel('Fraction complete')
    plt.text(0.05, 0.2, r'$\gamma=0.4$', fontsize=15)
    plt.title('AUC-proxy plot')
    if outpath != '':
        plt.savefig(outpath)
    plt.show()

def make_figure(filename,cluster=False,outpath=''):
    plt.ioff()
    mxs, cxs, mys, cys, dts =[], [], [], [], []
    for line in open(filename):
        if line.startswith('#'):
            continue
        desig, cx, mx, cy, my, dt = line.split()
        mxs.append(float(mx))
        cxs.append(float(cx))
        mys.append(float(my))
        cys.append(float(cy))
        dts.append(float(dt))

    fig=plt.figure(figsize=(18, 16))

    colormap = cm.inferno
    if cluster:
        colormap = mlc.ListedColormap ( np.random.rand ( 256,3))

    plt.quiver(cxs, cys, mxs, mys, dts, cmap=colormap,scale=0.3, width=0.0003)

    # plt.xlim(-0.2, 0.2)
    # plt.ylim(-0.2, 0.2)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    if outpath=='':
        outfile = filename+'.pdf'
    else:
        outfile = outpath +'.pdf'
    plt.savefig(outfile)
    plt.close()
    plt.ion()
