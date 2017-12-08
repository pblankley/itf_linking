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
import lib.MPC_library as MPC_library
import scipy.spatial
import pickle
from operator import add
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import util
import os
from clustering import train_clusters
Observatories = MPC_library.Observatories

ObservatoryXYZ = Observatories.ObservatoryXYZ

# Do the training run
# print('Starting training run:')
# for i,pix in enumerate(range(hp.nside2npix(nside))):
#     # Percent complete
#     out = i * 1. / len(range(hp.nside2npix(nside))) * 100
#     sys.stdout.write("\r%d%%" % out)
#     sys.stdout.flush()
#
#     pix_runs[pix] = find_clusters([pix], infilename, util.lunation_center(n), g_gdots=g_gdots, rtype='train')
#
# sys.stdout.write("\r%d%%" % 100)
# print('\n')
# #
# # # Save the results as a pickle
# with open(pickle_filename, 'wb') as handle:
#     pickle.dump(pix_runs, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print('Different dt values:',*list(pix_runs[0].keys()))


# GLOBALS
nside=8
#gs = [0.3, 0.35, 0.4, 0.45, 0.5]
gs = [0.4]
#gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
gdots = [-0.004, -0.002, 0.0, 0.002, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

# moons = [-11, -14, -17, -20, -23]
def tune(moons, nside, home_dir, g_gdots=g_gdots, dts=np.arange(5, 30, 5),
        radii=np.arange(0.0001, 0.0100, 0.0001), mincount=3):
    """ tuning docs"""
    abs_home_dir = os.path.abspath(home_dir)

    # Looping over five lunation centers, separated by 3 months each
    for i,n in enumerate(moons):
        # print('\n','{0}/{1}'.format(i,len(moons)),'\n')
        lunation = util.lunation_center(n)
        pix_runs = {}
        infilename=os.path.join(abs_home_dir, 'UnnObs_Training_1_line_A_%.1lf_pm15.0_r2.5.trans' % (lunation))
        pickle_filename = infilename.rstrip('trans') + 'train.pickle' # removed _v2 after train.

        for i,pix in enumerate(range(hp.nside2npix(nside))):
            # Percent complete
            # out = i * 1. / len(range(hp.nside2npix(nside))) * 100
            # sys.stdout.write("\r%d%%" % out)
            # sys.stdout.flush()

            # Do the training run
            pix_runs[pix] = train_clusters([pix], infilename, lunation_center(n), \
                                            g_gdots=g_gdots,dts=dts,radii=radii, mincount=mincount)

        # sys.stdout.write("\r%d%%" % 100)

        # Write the output to a pickle
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(pix_runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n')

    print('Find the best velocity / position scaling, our dt value.')

def plot_tune_results(moons, home_dir):
    """ only run after you have already run tune"""
    abs_home_dir = os.path.abspath(home_dir)

    for n in moons:
        infilename=os.path.join(abs_home_dir, 'UnnObs_Training_1_line_A_%.1lf_pm15.0_r2.5.trans' % (util.lunation_center(n)))
        pickle_filename = infilename.rstrip('trans') + 'train.pickle'

        with open(pickle_filename, 'rb') as handle:
            try:
                pix_runs = pickle.load(handle)
            except FileNotFoundError:
                raise FileNotFoundError('Cannot find this file. Hint: make sure you have run the tune() function first!')

            true_count_dict, mergedCounter_dict, mergedTime_dict=accessible_clusters(list(pix_runs.keys()), infilename=infilename)
            true_count=sum(true_count_dict.values())

            visual.number_clusters_plot(pix_runs,true_count)
            visual.number_errors_plot(pix_runs)
            visual.auc_plot(pix_runs,true_count)

def find_cluster_radius(moons, home_dir, dt, max_tol=1e-3):
    """ docs"""
    abs_home_dir = os.path.abspath(home_dir)

    print('Now that we have set dt={}, lets calculate the best cluster radius.'.format(dt))

    training_dict={}
    for n in moons:
        infilename=os.path.join(abs_home_dir, 'UnnObs_Training_1_line_A_%.1lf_pm15.0_r2.5.trans' % (util.lunation_center(n)))
        pickle_filename = infilename.rstrip('trans') + 'train.pickle'

        with open(pickle_filename, 'rb') as handle:
            pix_runs = pickle.load(handle)

            true_count_dict, mergedCounter_dict, mergedTime_dict=accessible_clusters(list(pix_runs.keys()), infilename=infilename)
            true_count=sum(true_count_dict.values())

            for i in range(99):
                errs=0
                clusts=0
                trues=0
                for pix in list(pix_runs.keys()):
                    nclusters = pix_runs[pixels[pix]][dt][1][i]
                    nerrors = pix_runs[pixels[pix]][dt][2][i]
                    ntrue = true_count_dict[pix]

                    errs += nerrors
                    clusts += nclusters
                    trues += ntrue
                if float(errs)/trues < max_tol:
                    print(i, pix_runs[pixels[pix]][dt][0][i], errs, clusts, trues)
                else:
                    training_dict[n] = pix_runs[pixels[pix]][dt][0][i], errs, clusts, trues
                    break

    cluster_radius = np.mean([v[0] for k, v in training_dict.items()])

    return cluster_radius
