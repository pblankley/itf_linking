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
from libr import MPC_library
import scipy.spatial
import pickle
from operator import add
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import util
import os
from clustering import train_clusters

# GLOBALS
nside=8
#gs = [0.3, 0.35, 0.4, 0.45, 0.5]
gs = [0.4]
#gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
gdots = [-0.004, -0.002, 0.0, 0.002, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

#########################################################################

def tune(moons, nside, home_dir, g_gdots=g_gdots, dts=np.arange(5, 30, 5),
        radii=np.arange(0.0001, 0.0100, 0.0001), mincount=3):
    """ This function takes in the g_gdots grid we iterate over, the dt's and radii we
    want to try, and the moons we want to include in the training run.  NOTE: it is
    not recommended to run this over the whole dataset, as that would be very slow,
    and likely not provide very exciting results. This function will take the values
    we pass and create pickle files for further analysis.
    ---------
    Args: moons; list, is a list of values of your choosing between -825 to 14,
            representing the different moons for lunation center. A recommended,
            relatively dense patch to start with is [-11, -14, -17, -20, -23].
          nside; int, the number of sides for the healpix dividing.
          home_dir; str, the path to the directory where we want to have the .trans
                        files, and where we want to output the pickles
          g_gdots; list of tuples of pairs, our grid of g and gdot.
          dts; array of floats to use to scale the velocity in relation to position
          radii; array of floats to use to search the radius of the given sizes in the
                    KD tree
          mincount; int, the minimum number of tracklets it takes to be considered a cluster.
    ---------
    Returns: None, writes pickles to given home_dir
    """
    abs_home_dir = os.path.abspath(home_dir)

    # Looping over five lunation centers, separated by 3 months each
    for i,n in enumerate(moons):
        lunation = util.lunation_center(n)
        pix_runs = {}
        infilename=os.path.join(abs_home_dir, 'UnnObs_Training_1_line_A_ec_%.1lf_pm15.0_r2.5.trans' % (lunation))
        pickle_filename = infilename.rstrip('trans') + 'train.pickle' # removed _v2 after train.

        for i,pix in enumerate(range(hp.nside2npix(nside))):
            # Do the training run
            pix_runs[pix] = train_clusters([pix], infilename, util.lunation_center(n), \
                                            g_gdots=g_gdots,dts=dts,radii=radii, mincount=mincount)

        # Write the output to a pickle
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(pix_runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Find the best velocity / position scaling, our dt value.')

def plot_tune_results(moons, home_dir):
    """ This function plots the related analysis plots of number of clusters,
    number of errors, and approx AUC.  NOTE: Only run this after you run tune()
    ---------
    Args: moons; list, is a list of values of your choosing between -825 to 14,
            representing the different moons for lunation center. A recommended,
            relatively dense patch to start with is [-11, -14, -17, -20, -23].
          home_dir; str, the path to the directory where we want to have the
            pickles from the tune() run.
    ---------
    Returns: None, plots the realted visualizations.
    """
    abs_home_dir = os.path.abspath(home_dir)

    for n in moons:
        infilename=os.path.join(abs_home_dir, 'UnnObs_Training_1_line_A_%.1lf_pm15.0_r2.5.trans' % (util.lunation_center(n)))
        pickle_filename = infilename.rstrip('trans') + 'train.pickle'

        if not os.path.isfile(pickle_filename):
            raise FileNotFoundError('Cannot find this file. Hint: make sure you have run the tune() function first!')
        with open(pickle_filename, 'rb') as handle:
            pix_runs = pickle.load(handle)

            true_count_dict, mergedCounter_dict, mergedTime_dict=accessible_clusters(list(pix_runs.keys()), infilename=infilename)
            true_count=sum(true_count_dict.values())

            visual.number_clusters_plot(pix_runs,true_count)
            visual.number_errors_plot(pix_runs)
            visual.auc_plot(pix_runs,true_count)

def find_cluster_radius(moons, home_dir, dt, max_tol=1e-3):
    """ This function finds the optimal cluster radius, given a value for dt
    and a maximum tolerable error rate.  The max error rate defaults to 0.1%.
    ---------
    Args: moons; list, is a list of values of your choosing between -825 to 14,
            representing the different moons for lunation center. A recommended,
            relatively dense patch to start with is [-11, -14, -17, -20, -23].
          home_dir; str, the path to the directory where we want to have the
            pickles from the tune() run.
          dt; float, the dt you decided to use based on the previous plots, or
            subject matter knowledge.
          max_tol; float, the maximum realtive error we tolerate in our output.
            defaults to 1e-3 or 0.1%
    ---------
    Returns: float, the optimal cluster radius (for finding the most clusters),
                while remaining under the specified error rate.
    """
    abs_home_dir = os.path.abspath(home_dir)

    print('Now that we have set dt={}, lets calculate the best cluster radius.'.format(dt))

    training_dict={}
    for n in moons:
        infilename=os.path.join(abs_home_dir, 'UnnObs_Training_1_line_A_ec_%.1lf_pm15.0_r2.5.trans' % (util.lunation_center(n)))
        pickle_filename = infilename.rstrip('trans') + 'train.pickle'

        if not os.path.isfile(pickle_filename):
            raise FileNotFoundError('Cannot find this file. Hint: make sure you have run the tune() function first!')
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
