# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import util
import visual
import pickle
import healpy as hp
from collections import Counter
from lib import MPC_library
from clustering import find_clusters, generate_sky_region_files, accessible_clusters
from clustering import train_clusters, test_clusters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# os.path.join(BASE_DIR, 'data/plist_df.json')

Observatories = MPC_library.Observatories

###############################################################################

#gs = [0.3, 0.35, 0.4, 0.45, 0.5]
gs = [0.4]
#gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
gdots = [-0.004, -0.002, 0.0, 0.002, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

pix_runs = {}
nside=8
n = -14
dt=15.0
cr=0.00124

infilename='demo_train/UnnObs_Training_1_line_A_ec_labelled_2457308.5_pm15.0_r2.5.trans'
pickle_filename = infilename.rstrip('.trans') + '.pickle'


# with open(pickle_filename, 'rb') as handle:
#     pix_runs = pickle.load(handle)
#
# true_count_dict, mergedCounter_dict, mergedTime_dict = accessible_clusters(list(pix_runs.keys()), infilename=infilename)
# true_count = sum(true_count_dict.values())
# print('True count of clusters: {}'.format(true_count))
#
# # Plots
# visual.number_clusters_plot(pix_runs,true_count)
# visual.number_errors_plot(pix_runs)
# visual.auc_plot(pix_runs,true_count)

# Actually, this should be related to the angle between the direction of
# Earth's motion and the direction of the observations.

print('Based on our tuning, the best dt is {0} and best cluster radius is {1}'.format(dt,cr))

pixels=range(hp.nside2npix(nside))
# right, wrong, ids_right, ids_wrong = test_clusters(pixels, infilename, util.lunation_center(n), \
#                                              mincount=3, dt=15.0,rad= 0.00124)
#
# print('Using our optimal parameters we got {0} percent of clusters with {1} errors.'.format(right/true_count,wrong))
# print('We got',right,'right and',wrong,'wrong out of total',true_count)



print('Now that we have used training data to come up with dt and cluster radius, lets run on the ITF.')

# luination center for itf -14 ....... 2457308.5
itf_file = 'demo_itf/itf_new_1_line_2457308.5_pm15.0_r2.5.trans'
itf_pickle = itf_file.rstrip('trans') + 'test.pickle'
itf_n = -14
itf_nside = 8
itf_t_ref = util.lunation_center(itf_n)

pixels = range(hp.nside2npix(itf_nside))
gs =[0.4]
gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

# TODO activate
# print('Starting ITF run...')
# itf_raw_results, itf_clust_id_dict = find_clusters(pixels, itf_file, itf_t_ref, g_gdots=g_gdots,dt=dt,rad=cr)
# print('ITF run finished!')
#
# with open(itf_pickle, 'wb') as handle:
#         pickle.dump(itf_raw_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # get results back
# with open(itf_pickle, 'rb') as handle:
#         itf_raw_results = pickle.load(handle)

# itf_tracklets_dict = util.get_original_tracklets_dict('demo_reference/itf_new_1_line_subset.mpc')
# itf_obs_array = util.get_original_observation_array('demo_reference/itf_new_1_line_subset.txt')

# obs_dict={}
# for cluster_key in itf_raw_results.keys():
#     obs_dict[cluster_key] = util.get_observations(cluster_key, itf_tracklets_dict, itf_obs_array)
#
# with open('demo_itf/itf_results', 'wb') as handle:
    # pickle.dump(obs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# get results back
# with open('demo_itf/itf_results', 'rb') as handle:
#     itf_results = pickle.load(handle)

# print('We clustered {0} asteroids, with a total of {1} tracklets clustered!'.format( \
#                     len(itf_results.keys()),sum(len(v) for v in itf_results.values())))


c_key = 'P10oagN|P10oahr|P10ohw2'
# print('Example output for one cluster:','\n','Key:,',c_key,'\n','Output',itf_results[c_key])
# TODO make a function that generates one sky region file

# generate_sky_region_files('demo_itf/itf_new_1_line_2457308.5_pm15.0_r2.5.trans', nside=nside, n=-14)

# visual.make_figure('demo_itf/itf_new_1_line_2457308.5_pm15.0_r2.5_hp_017_g0.40_gdot+0.0e+00')

# pull in the orbit fitting results then plot revised with orbit fitting  (drop non-members)

# after fitting orbits, we have found these asteroids.

# whole run is -825 to 10
# TODO activate above this

# gs = [0.4]
# gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
# g_gdots = [(x,y) for x in gs for y in gdots]
#
pixels= range(hp.nside2npix(nside))
#
real_run, clust_ids = find_clusters(pixels, infilename,  util.lunation_center(n), g_gdots=g_gdots, dt=15)
print(clust_ids)
# print('How many in the run? There were {}.'.format(len(real_run)))
#
# with open('demo_data/demo_results.pickle', 'wb') as handle:
#     pickle.dump(real_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
generate_sky_region_files(infilename, [281], nside=nside, n=n)
generate_sky_region_files(infilename, [281], nside=nside, n=n, cluster_id_dict=clust_ids)

visual.make_figure('demo_train/UnnObs_Training_1_line_A_ec_labelled_2457308.5_pm15.0_r2.5_hp_281_g0.40_gdot+0.0e+00')
visual.make_figure('demo_train/UnnObs_Training_1_line_A_ec_labelled_2457308.5_pm15.0_r2.5_hp_281_g0.40_gdot+0.0e+00_cid',cluster=True)
 # visual.make_figure(path_makefig)
# dne = []
# for p in range(hp.nside2npix(nside)):
#     path_makefig = 'demo_itf/itf_new_1_line_2457308.5_pm15.0_r2.5_hp_{}_g0.40_gdot+0.0e+00'.format(('%03d' % (p)))
#     try:
#         visual.make_figure(path_makefig)
#     except:
#         dne.append(p)
#         pass
# print('There were {} that had no clusters.'.format(len(dne)))

# dne = []
# for p in range(hp.nside2npix(nside)):
#     path_makefig = 'demo_train/UnnObs_Training_1_line_A_ec_labelled_2457308.5_pm15.0_r2.5_hp_{}_g0.40_gdot+0.0e+00'.format(('%03d' % (p)))
#     try:
#         visual.make_figure(path_makefig)
#     except:
#         dne.append(p)
#         pass
# print('There were {} that had no clusters.'.format(len(dne)))

#############################################################################


# earth_vec = Observatories.getObservatoryPosition('500', util.lunation_center(n))
# pixels, infilename, t_ref,g_gdots=g_gdots, mincount=3,dt=15,rad=0.00124
# errs, clusts, trues = util.do_test_run(pix_runs, true_count_dict, earth_vec, dt=15, nside=nside)
