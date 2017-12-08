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
from libr import MPC_library
from clustering import find_clusters, generate_sky_region_files, accessible_clusters
from clustering import train_clusters, test_clusters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global parameters
gs = [0.4]
gdots = [-0.004, -0.002, 0.0, 0.002, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

nside=8
n = -14
dt=15.0
cr=0.00124
pixels=range(hp.nside2npix(nside))

infilename=os.path.join(BASE_DIR, 'demo_train/UnnObs_Training_1_line_A_ec_labelled_2457308.5_pm15.0_r2.5.trans')

print('Based on our tuning, the best dt is {0} and best cluster radius is {1}'.format(dt,cr))

true_count_set, mergedCounter_dict, mergedTime_dict = accessible_clusters(pixels, infilename=infilename)
true_count = len(true_count_set)
print('True count of clusters: {}'.format(true_count))

right, wrong, ids_right, ids_wrong = test_clusters(pixels, infilename, util.lunation_center(n), \
                                                    dt=dt,rad=cr)

print('Using our optimal parameters we got {0} percent of clusters with {1} percent errors.'.format(right/true_count,wrong/true_count))
print('We got',right,'right and',wrong,'wrong out of total',true_count)

print('Now that we have shown our performance on training data, lets run on the ITF.')

itf_file = os.path.join(BASE_DIR, 'demo_itf/itf_new_1_line_ec_2457308.5_pm15.0_r2.5.trans')
itf_pickle = itf_file.rstrip('.trans') + '.pickle'
itf_n = -14
itf_nside = 8
itf_pixels = range(hp.nside2npix(itf_nside))

itf_raw_results, itf_clust_ids = find_clusters(pixels, itf_file, util.lunation_center(itf_n), g_gdots=g_gdots,dt=dt,rad=cr)

check_mpc = os.path.join(BASE_DIR, 'demo_itf/itf_new_1_line_ec_subset.mpc')
check_txt = os.path.join(BASE_DIR, 'demo_itf/itf_new_1_line_subset.txt')
itf_tracklets_dict = util.get_original_tracklets_dict(check_mpc)
itf_obs_array = util.get_original_observation_array(check_txt)

obs_dict={}
for cluster_key in itf_raw_results.keys():
    obs_dict[cluster_key] = util.get_observations(cluster_key, itf_tracklets_dict, itf_obs_array)

print('We clustered {0} asteroids from the ITF!'.format(len(obs_dict.keys())))
print('NOTE: We only count clusters with 3 or more tracklets.')

generate_sky_region_files(itf_file, [281], nside=itf_nside, n=itf_n)
generate_sky_region_files(itf_file, [281], nside=itf_nside, n=itf_n, cluster_id_dict=itf_clust_ids)

dne = []
for p in [281]:
    path_makefig = 'demo_itf/itf_new_1_line_ec_2457308.5_pm15.0_r2.5_hp_{}_g0.40_gdot+0.0e+00'.format(('%03d' % (p)))
    path_cid = 'demo_itf/itf_new_1_line_ec_2457308.5_pm15.0_r2.5_hp_{}_g0.40_gdot+0.0e+00_cid'.format(('%03d' % (p)))
    path_makefig = os.path.join(BASE_DIR, path_makefig)
    path_cid = os.path.join(BASE_DIR, path_cid)
    try:
        visual.make_figure(path_makefig,outpath=os.path.join(BASE_DIR,'figure_colored_by_time_all_tracklets'))
        visual.make_figure(path_cid,cluster=True,outpath=os.path.join(BASE_DIR,'figure_colored_by_cluster_only_clustered_tracklets'))
    except:
        dne.append(p)
        pass

c_key = 'P10oagN|P10oahr|P10ohw2'
print('Example output for one cluster:','\n','Key:,',c_key,'\n','Output',obs_dict[c_key])

print('\n')
print('Check out the pdf result in the current directory!')
