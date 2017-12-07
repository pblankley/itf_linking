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
n = -11
dt=15.0
cr=0.00124

infilename='demo_data/UnnObs_Training_1_line_A_2457397.5_pm15.0_r2.5.trans'
pickle_filename = infilename.rstrip('.trans') + '.pickle'

# Do the training run
# print('Starting training run:')
# for i,pix in enumerate(range(hp.nside2npix(nside))):
#     # Percent complete
#     out = i * 1. / len(range(hp.nside2npix(nside))) * 100
#     sys.stdout.write("\r%d%%" % out)
#     sys.stdout.flush()
#
#     pix_runs[pix] = util.do_training_run([pix], infilename, util.lunation_center(n), g_gdots=g_gdots)
#
# sys.stdout.write("\r%d%%" % 100)
# print('\n')
#
# # Save the results as a pickle
# with open(pickle_filename, 'wb') as handle:
#     pickle.dump(pix_runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(pickle_filename, 'rb') as handle:
    pix_runs = pickle.load(handle)

print('Different dt values:',*list(pix_runs[0].keys()))
# raise ValueError('stop')

# Should save these results in files
# true_count_dict, mergedCounter_dict, mergedTime_dict = util.accessible_clusters(list(pix_runs.keys()), infilename=infilename)
# true_count = sum(true_count_dict.values())
# print('True count of clusters: {}'.format(true_count))

# num_occ=Counter()
# for pix in range(hp.nside2npix(nside)):
#     hist=util.make_histogram(mergedCounter_dict[pix])
#     for k, v in hist.items():
#         num_occ.update({k: len(v)})
#
# print(num_occ)

# visual.number_clusters_plot(pix_runs,true_count)
# visual.number_errors_plot(pix_runs)
# visual.auc_plot(pix_runs,true_count)

# Actually, this should be related to the angle between the direction of
# Earth's motion and the direction of the observations.

print('Based on our tuning, the best dt is {0} and best cluster radius is {1}'.format(dt,cr))
# earth_vec = Observatories.getObservatoryPosition('500', util.lunation_center(n))
# pixels, infilename, t_ref,g_gdots=g_gdots, mincount=3,dt=15,rad=0.00124
# errs, clusts, trues = util.do_test_run(pix_runs, true_count_dict, earth_vec, dt=15, nside=nside)
pixels=range(hp.nside2npix(nside))

# right, wrong, ids_right, ids_wrong = util.do_test_run(pixels, infilename, util.lunation_center(n), mincount=3, dt=15.0,rad= 0.00124)

# print('Using our optimal parameters we got {0} percent of clusters with {1} errors.'.format(right/(right+wrong),wrong))

# print('errors:',errs,'clusts:',clusts,'true count of clusters:',trues)
# print('We acheived {0} percent accuracy with {1} errors.'.format(clusts/trues,errs))


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

# print('Starting ITF run...')
# itf_raw_results = util.do_run(pixels, itf_file, itf_t_ref,g_gdots=g_gdots,dt=dt,rad=cr)
# print('ITF run finished!')
#
# with open(itf_pickle, 'wb') as handle:
#         pickle.dump(itf_raw_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # get results back
# with open(itf_pickle, 'rb') as handle:
#         itf_raw_results = pickle.load(handle)

# Get the output in a format for the MPC to check with orbit fitting
itf_tracklets_dict = util.get_original_tracklets_dict('demo_reference/itf_new_1_line_subset.mpc')
itf_obs_array = util.get_original_observation_array('demo_reference/itf_new_1_line_subset.txt')

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

# util.generate_sky_region_files('demo_itf/itf_new_1_line_2457308.5_pm15.0_r2.5.trans', nside=8, n=-11)
# visual.make_figure('demo_data/UnnObs_Training_1_line_A_2457397.5_pm15.0_r2.5_hp_023_g0.40_gdot+0.0e+00')
# visual.make_figure('demo_itf/itf_new_1_line_2457308.5_pm15.0_r2.5_hp_017_g0.40_gdot+0.0e+00')

# pull in the orbit fitting results then plot revised with orbit fitting  (drop non-members)

# after fitting orbits, we have found these asteroids.

# whole run is -825 to 10

# gs = [0.4]
# gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
# g_gdots = [(x,y) for x in gs for y in gdots]
#
# pixels= range(hp.nside2npix(nside))
#
# real_run = util.do_run(g_gdots, pixels, infilename, nside=nside, n=n, dt=15)
#
# print('How many in the run? There were {}.'.format(len(real_run)))
#
# with open('demo_data/demo_results.pickle', 'wb') as handle:
#     pickle.dump(real_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
util.generate_sky_region_files(infilename, nside=nside, n=n)
#
dne = []
for p in range(hp.nside2npix(nside)):
    path_makefig = 'demo_itf/itf_new_1_line_2457308.5_pm15.0_r2.5_hp_{}_g0.40_gdot+0.0e+00'.format(('%03d' % (p)))
    try:
        visual.make_figure(path_makefig)
    except:
        dne.append(p)
        pass
print('There were {} that had no clusters.'.format(len(dne)))

dne = []
for p in range(hp.nside2npix(nside)):
    path_makefig = 'demo_data/UnnObs_Training_1_line_A_2457397.5_pm15.0_r2.5_hp_{}_g0.40_gdot+0.0e+00'.format(('%03d' % (p)))
    try:
        visual.make_figure(path_makefig)
    except:
        dne.append(p)
        pass
print('There were {} that had no clusters.'.format(len(dne)))

#############################################################################
