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
from matplotlib.colors import Normalize


def number_clusters_plot(pix_runs,true_count):
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
    #plt.text(0.005, 400, r'$\gamma=0.4$', fontsize=15)
    plt.legend()
    plt.title('Number of clusters by cluster radius.')
    plt.savefig('demo_data/nclusters_demo')
    plt.show()


def number_errors_plot(pix_runs):
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
        '''
        if dt==50:
            for d, nc, ne in zip(ds, nclusters, nerrors):
                print(d, nc, ne)
        '''


        plt.plot(ds, nerrors, label=dt)

    plt.xscale("log", nonposx='clip')
    plt.ylim((0,3000))
    plt.xlabel('d (cluster radius)')
    plt.ylabel('N errors')
    plt.text(0.0005, 1000, r'$\gamma=0.4$', fontsize=15)
    plt.legend()
    plt.title('Number of errors by cluster radius')
    plt.savefig('demo_data/nerrors_demo')
    plt.show()


def auc_plot(pix_runs,true_count):
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
    #plt.legend()
    plt.title('AUC-proxy plot')
    plt.savefig('demo_data/AUC_demo')
    plt.show()

def make_figure(filename,outpath=''):
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

    #norm = Normalize()
    #norm.autoscale(colors)

    colormap = cm.inferno

    plt.quiver(cxs, cys, mxs, mys, dts, scale=0.3, width=0.0003)

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
