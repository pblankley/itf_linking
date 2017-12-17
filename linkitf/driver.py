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
import cleaning as cl
from clustering import find_clusters, cluster_clusters, postprocessing, cluster_months
from itf_clean import clean_itf_data_mpc, clean_training_data_mpc
import os
import sys

######################### FUNCTION DEFINITIONS ##################################

# Put the runs in function so they do not automatically execute
def run_itf(path_to_itf,pixels,g_gdots,dt,cr):
    """ Run the whole ITF file """
    home_dir = os.path.dirname(path_to_itf)
    print('Starting run...')
    for prog,n in enumerate(range(-825,14)):
        # Percent complete
        out = prog * 1. / len(range(-825,14)) * 100
        sys.stdout.write("\r%d%%" % out)
        sys.stdout.flush()

        itf_file = os.path.join(home_dir, 'itf_new_1_line_ec_{}_pm15.0_r2.5.trans'.format(str(util.lunation_center(n))))

        if os.path.isfile(itf_file):
            itf_raw_results, itf_clust_ids = find_clusters(pixels, itf_file, util.lunation_center(n), g_gdots=g_gdots,dt=dt,rad=cr)

            # itf_tracklets_dict = util.get_original_tracklets_dict(os.path.join(mpc_path))
            # itf_obs_array = util.get_original_observation_array(os.path.join(txt_path))
            #
            # obs_dict={}
            # for cluster_key in itf_raw_results.keys():
            #     obs_dict[cluster_key] = util.get_observations(cluster_key, itf_tracklets_dict, itf_obs_array)

            with open(os.path.join(home_dir,'itf_result_{}_initial.pickle'.format(str(util.lunation_center(n)))),'wb') as handle:
                pickle.dump(itf_raw_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stdout.write("\r%d%%" % 100)
    print('\n')
    print('Run finished!')

def cluster_clusters_itf(path_to_itf,pixels,nside,dt,cr,new_rad):
    home_dir = os.path.dirname(path_to_itf)
    print('Starting cluster clusters run...')
    for prog, n in enumerate(range(-825,14)):
        # Percent complete
        out = prog * 1. / len(range(-825,14)) * 100
        sys.stdout.write("\r%d%%" % out)
        sys.stdout.flush()

        itf_file = os.path.join(home_dir, 'itf_new_1_line_ec_{}_pm15.0_r2.5.trans'.format(str(util.lunation_center(n))))

        if os.path.isfile(itf_file):
            # Get the previously calc'd result
            with open(os.path.join(home_dir,'itf_result_{}_initial.pickle'.format(str(util.lunation_center(n)))),'rb') as handle:
                clust_counter = pickle.load(handle)

            coc_counter, coc_ids = cluster_clusters(itf_file, clust_counter, pixels, nside, n, dt=dt, rad=cr, \
                                                                            new_rad=new_rad, gi=0.4, gdoti=0.0)

            with open(os.path.join(home_dir,'itf_result_{}_coc.pickle'.format(str(util.lunation_center(n)))),'wb') as handle:
                pickle.dump(coc_counter, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stdout.write("\r%d%%" % 100)
    print('\n')
    print('Run finished!')

def postprocessing_itf(path_to_itf,pixels,nside):
    home_dir = os.path.dirname(path_to_itf)
    print('Starting postprocessing run...')
    for prog, n in enumerate(range(-825,14)):
        # Percent complete
        out = prog * 1. / len(range(-825,14)) * 100
        sys.stdout.write("\r%d%%" % out)
        sys.stdout.flush()

        itf_file = os.path.join(home_dir, 'itf_new_1_line_ec_{}_pm15.0_r2.5.trans'.format(str(util.lunation_center(n))))

        if os.path.isfile(itf_file):
            # Get the previously calc'd result
            with open(os.path.join(home_dir,'itf_result_{}_coc.pickle'.format(str(util.lunation_center(n)))),'rb') as handle:
                clust_counter = pickle.load(handle)

            fit_dict = postprocessing(itf_file, clust_counter, pixels, nside, n, orb_elms=True, gi=0.4, gdoti=0.0)

            with open(os.path.join(home_dir,'itf_result_{}_orbelem.pickle'.format(str(util.lunation_center(n)))),'wb') as handle:
                pickle.dump(fit_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stdout.write("\r%d%%" % 100)
    print('\n')
    print('Run finished!')

def cluster_month_over_month_itf(path_to_itf,rad):
    home_dir = os.path.dirname(path_to_itf)
    fit_dicts = []
    print('Starting month over month cluster run...')
    for prog, n in enumerate(range(-825,14)):

        itf_file = os.path.join(home_dir, 'itf_new_1_line_ec_{}_pm15.0_r2.5.trans'.format(str(util.lunation_center(n))))

        if os.path.isfile(itf_file):
            # Get the previously calc'd result
            with open(os.path.join(home_dir,'itf_result_{}_orbelem.pickle'.format(str(util.lunation_center(n)))),'rb') as handle:
                fit_dict = pickle.load(handle)
            fit_dicts.append(fit_dict)

    print('Data loaded...')
    final_dict, final_dict_cid = cluster_months(fit_dicts,rad=rad)

    with open(os.path.join(home_dir,'itf_final_results.pickle'),'wb') as handle:
        pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(home_dir,'itf_final_results_cid.pickle'),'wb') as handle:
        pickle.dump(final_dict_cid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Run finished!')

###############################################################################

######################## TRAINING FUNCTIONS ####################################
def run_train(path_to_train,pixels,g_gdots,dt,cr):
    """ Run the whole Training file """
    home_dir = os.path.dirname(path_to_train)
    print('Starting initial run...')

    for prog, n in enumerate(range(-825,14)):
        # Percent complete
        out = prog * 1. / len(range(-825,14)) * 100
        sys.stdout.write("\r%d%%" % out)
        sys.stdout.flush()

        train_file = os.path.join(home_dir, 'UnnObs_Training_1_line_A_ec_{}_pm15.0_r2.5.trans'.format(str(util.lunation_center(n))))

        if os.path.isfile(train_file):
            train_raw_results, train_clust_ids = find_clusters(pixels, train_file, util.lunation_center(n), g_gdots=g_gdots,dt=dt,rad=cr)

            with open(os.path.join(home_dir,'train_result_{}_initial.pickle'.format(str(util.lunation_center(n)))),'wb') as handle:
                pickle.dump(train_raw_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stdout.write("\r%d%%" % 100)
    print('\n')
    print('Run finished!')

def cluster_clusters_train(path_to_train,pixels,nside,dt,cr,new_rad):
    home_dir = os.path.dirname(path_to_train)
    print('Starting cluster clusters run...')
    for prog, n in enumerate(range(-825,14)):
        # Percent complete
        out = prog * 1. / len(range(-825,14)) * 100
        sys.stdout.write("\r%d%%" % out)
        sys.stdout.flush()

        train_file = os.path.join(home_dir, 'UnnObs_Training_1_line_A_ec_{}_pm15.0_r2.5.trans'.format(str(util.lunation_center(n))))

        if os.path.isfile(train_file):
            # Get the previously calc'd result
            with open(os.path.join(home_dir,'train_result_{}_initial.pickle'.format(str(util.lunation_center(n)))),'rb') as handle:
                clust_counter = pickle.load(handle)

            coc_counter, coc_ids = cluster_clusters(train_file, clust_counter, pixels, nside, n, dt=dt, rad=cr, \
                                                                            new_rad=new_rad, gi=0.4, gdoti=0.0)

            with open(os.path.join(home_dir,'train_result_{}_coc.pickle'.format(str(util.lunation_center(n)))),'wb') as handle:
                pickle.dump(coc_counter, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stdout.write("\r%d%%" % 100)
    print('\n')
    print('Run finished!')

def postprocessing_train(path_to_train,pixels,nside):
    home_dir = os.path.dirname(path_to_train)
    print('Starting postprocessing run...')
    for prog, n in enumerate(range(-825,14)):
        # Percent complete
        out = prog * 1. / len(range(-825,14)) * 100
        sys.stdout.write("\r%d%%" % out)
        sys.stdout.flush()

        train_file = os.path.join(home_dir, 'UnnObs_Training_1_line_A_ec_{}_pm15.0_r2.5.trans'.format(str(util.lunation_center(n))))

        if os.path.isfile(train_file):
            # Get the previously calc'd result
            with open(os.path.join(home_dir,'train_result_{}_coc.pickle'.format(str(util.lunation_center(n)))),'rb') as handle:
                clust_counter = pickle.load(handle)

            fit_dict = postprocessing(train_file, clust_counter, pixels, nside, n, orb_elms=True, gi=0.4, gdoti=0.0)

            with open(os.path.join(home_dir,'train_result_{}_orbelem.pickle'.format(str(util.lunation_center(n)))),'wb') as handle:
                pickle.dump(fit_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stdout.write("\r%d%%" % 100)
    print('\n')
    print('Run finished!')

def cluster_month_over_month_train(path_to_train,rad):
    home_dir = os.path.dirname(path_to_train)
    fit_dicts = []
    print('Starting month over month cluster run...')
    for prog, n in enumerate(range(-825,14)):

        train_file = os.path.join(home_dir, 'UnnObs_Training_1_line_A_ec_{}_pm15.0_r2.5.trans'.format(str(util.lunation_center(n))))

        if os.path.isfile(train_file):
            # Get the previously calc'd result
            with open(os.path.join(home_dir,'train_result_{}_orbelem.pickle'.format(str(util.lunation_center(n)))),'rb') as handle:
                fit_dict = pickle.load(handle)
            fit_dicts.append(fit_dict)

    print('Data loaded...')
    final_dict, final_dict_cid = cluster_months(fit_dicts,rad=rad)

    with open(os.path.join(home_dir,'train_final_results.pickle'),'wb') as handle:
        pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(home_dir,'train_final_results_cid.pickle'),'wb') as handle:
        pickle.dump(final_dict_cid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Run finished!')

##########################################################################

"""
****************************** INSTRUCTIONS ***************************
Below you can first define the correct paths and vairables you would like
to use.  If you want to replicate our results, stick with the defaults below.

Since each call of the functions below (listed under their correct headers)
are quite time intensive, we have commented them out.  When you want to run one or
more of the sections of the driver, just uncomment the line realted to what you want
to run, and run the file.

If you are starting with nothing but the main file, enter the path below for either
training or ITF or both. Then if you are running just the training file, uncomment
each line in TRAINING SECTION that is commented out under its descriptive header.
If you are running just the ITF file, uncomment each line in ITF SECTION that
is commented out under its descriptive header.  If you are running both files,
uncomment every line in both the TRAINING and ITF sections.
"""
if __name__=='__main__':

    ########################### USER INPUT #############################
    mpcpath_train = 'data/train/UnnObs_Training_1_line_A_ec.mpc'
    mpcpath_itf = 'data/itf/itf_new_1_line_ec.mpc'
    ####################################################################

    # Define the variables to use
    gs = [0.4]
    gdots = [-0.004, -0.002, 0.0, 0.002, 0.004]
    g_gdots = [(x,y) for x in gs for y in gdots]
    nside=8
    dt=15.0
    cr=0.00124
    meta_rad=.05
    pixels = range(hp.nside2npix(nside))
    # txtpath = 'data/itf/itf_new_1_line.txt'

    # Get abs paths
    mpc_path = os.path.abspath(mpcpath_itf)
    mpc_path_train = os.path.abspath(mpcpath_train)
    # txt_path = os.path.abspath(txtpath)

    ###################### TRAINING SECTION ############################
    """Clean the training data (one-time run)"""
    # clean_training_data_mpc(mpc_path_train)

    """Run the training file (saves pickles)"""
    # run_train(mpc_path_train,pixels,g_gdots,dt,cr)

    """Cluster the clusters (saves pickles)"""
    # cluster_clusters_train(mpc_path_train,pixels,nside,dt,cr,new_rad=cr)

    """Postprocessing (saves pickles)"""
    # postprocessing_train(mpc_path_train,pixels,nside)

    """Cluster month over month (saves pickles)"""
    # cluster_month_over_month_train(mpc_path_train,rad=meta_rad)


    ####################### ITF SECTION ###############################
    """Clean the itf data (one-time run)"""
    # clean_itf_data_mpc(mpc_path)

    """Run the itf  (saves pickles)"""
    # run_itf(mpc_path,pixels,g_gdots,dt,cr)

    """Cluster the clusters  (saves pickles)"""
    # cluster_clusters_itf(mpc_path,pixels,nside,dt,cr,new_rad=cr)

    """Postprocessing  (saves pickles)"""
    # postprocessing_train(mpc_path,pixels,nside)

    """Meta clustering month over month"""
    # cluster_month_over_month_train(mpc_path,rad=meta_rad)


    ############################## END #####################################
