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

Observatories = MPC_library.Observatories

ObservatoryXYZ = Observatories.ObservatoryXYZ

# GLOBALS
nside=8
#gs = [0.3, 0.35, 0.4, 0.45, 0.5]
gs = [0.4]
#gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
gdots = [-0.004, -0.002, 0.0, 0.002, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

# moons = [-11, -14, -17, -20, -23]
def tune(moons):
    # Looping over five lunation centers, separated by 3 months each
    for i,n in enumerate(moons):
        print('\n','{0}/{1}'.format(i,len(moons)),'\n')
        pix_runs = {}
        infilename='data/UnnObs_Training_1_line_A_%.1lf_pm15.0_r2.5.trans' % (lunation_center(n))
        pickle_filename = infilename.rstrip('trans') + 'train.v2_pickle'

        for i,pix in enumerate(range(hp.nside2npix(nside))):
            # Percent complete
            out = i * 1. / len(range(hp.nside2npix(nside))) * 100
            sys.stdout.write("\r%d%%" % out)
            sys.stdout.flush()

            # Do the training run
            pix_runs[pix] = util.do_training_run([pix], infilename, lunation_center(n), g_gdots=g_gdots)
        sys.stdout.write("\r%d%%" % 100)

        # Write the output to a pickle
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(pix_runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n')
    # Do the training run
    print('Starting training run:')
    for i,pix in enumerate(range(hp.nside2npix(nside))):
        # Percent complete
        out = i * 1. / len(range(hp.nside2npix(nside))) * 100
        sys.stdout.write("\r%d%%" % out)
        sys.stdout.flush()

        pix_runs[pix] = util.do_training_run([pix], infilename, util.lunation_center(n), g_gdots=g_gdots)

    sys.stdout.write("\r%d%%" % out)

    print('Find the best velocity / position scaling, our dt value.')

    for n in moons:
        infilename='data/UnnObs_Training_1_line_A_%.1lf_pm15.0_r2.5.trans' % (lunation_center(n))
        pickle_filename = infilename.rstrip('trans') + 'v2_pickle'
        dt=15.
        with open(pickle_filename, 'rb') as handle:
            pix_runs = pickle.load(handle)

            # Should save these results in files
            true_count_dict, mergedCounter_dict, mergedTime_dict=accessible_clusters(list(pix_runs.keys()), infilename=infilename)
            true_count=sum(true_count_dict.values())

            visual.number_clusters_plot(pix_runs,true_count)
            visual.number_errors_plot(pix_runs)
            visual.auc_plot(pix_runs,true_count)

    print("Based on these plots and Matt's subject matter knowledge, we choose dt to be 15.")

    print('Now that we have dt=15, lets calculate the best cluster radius.')

    error_rate_limit=1e-3
    training_dict={}
    for n in moons:
        infilename='data/UnnObs_Training_1_line_A_%.1lf_pm15.0_r2.5.trans' % (lunation_center(n))
        pickle_filename = infilename.rstrip('trans') + 'v2_pickle'
        dt=15.
        with open(pickle_filename, 'rb') as handle:
            pix_runs = pickle.load(handle)

            # Should save these results in files
            true_count_dict, mergedCounter_dict, mergedTime_dict=accessible_clusters(list(pix_runs.keys()), infilename=infilename)
            true_count=sum(true_count_dict.values())
            true_count

            print(n)
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
                if float(errs)/trues<error_rate_limit:
                    print(i, pix_runs[pixels[pix]][dt][0][i], errs, clusts, trues)
                else:
                    training_dict[n] = pix_runs[pixels[pix]][dt][0][i], errs, clusts, trues
                    break

    cluster_radius = np.mean([v[0] for k, v in training_dict.items()])

    print('The best cluster radius is:',cluster_radius)
