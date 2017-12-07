# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import util  # Get the module structured so this works.
import pickle
import healpy as hp
from collections import Counter
import MPC_library
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

infilename='demo_data/UnnObs_Training_1_line_A_2457397.5_pm15.0_r2.5.trans'
pickle_filename = infilename.rstrip('trans') + '.pickle'

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
# sys.stdout.write("\r%d%%" % out)
# print('\n')

# Save the results as a pickle
# with open(pickle_filename, 'wb') as handle:
#     pickle.dump(pix_runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(pickle_filename, 'rb') as handle:
    pix_runs = pickle.load(handle)

print('Different dt values:',*list(pix_runs[0].keys()))
# raise ValueError('stop')
# Should save these results in files
true_count_dict, mergedCounter_dict, mergedTime_dict = util.accessible_clusters(list(pix_runs.keys()), infilename=infilename)
true_count = sum(true_count_dict.values())
print('True count of clusters: {}'.format(true_count))

# num_occ=Counter()
# for pix in range(hp.nside2npix(nside)):
#     hist=util.make_histogram(mergedCounter_dict[pix])
#     for k, v in hist.items():
#         num_occ.update({k: len(v)})
#
# print(num_occ)

util.number_clusters_plot(pix_runs,true_count)
util.number_errors_plot(pix_runs)
util.auc_plot(pix_runs,true_count)

# Actually, this should be related to the angle between the direction of
# Earth's motion and the direction of the observations.
earth_vec = Observatories.getObservatoryPosition('500', util.lunation_center(n))

errs, clusts, trues = util.evaluate(pix_runs, true_count_dict, earth_vec, dt=15, nside=nside)

print('errors:',errs,'clusts:',clusts,'true count of clusters:',trues)
print('We acheived {0} percent accuracy with {1} errors.'.format(clusts/trues,errs))

# Show the dt and cluster radius we decide on

# now we are doing a run over an itf time slice
# make the dictionary for output

# present a graph of what we think

# pull in the orbit fitting results then plot revised with orbit fitting  (drop non-members)

# after fitting orbits, we have found these asteroids.

# whole run is -825 to 10

gs = [0.4]
gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

pixels= range(hp.nside2npix(nside))

real_run = util.do_run(g_gdots, pixels, infilename, nside=nside, n=n, dt=15)

print('How many in the run? There were {}.'.format(len(real_run)))

with open('demo_data/demo_results.pickle', 'wb') as handle:
    pickle.dump(real_run, handle, protocol=pickle.HIGHEST_PROTOCOL)

util.generate_sky_region_files(infilename, nside=nside, n=n)

for p in range(hp.nside2npix(nside)):
    path_makefig = 'demo_data/UnnObs_Training_1_line_A_2457397.5_pm15.0_r2.5_hp_{}_g0.40_gdot+0.0e+00'.format(p)
    util.make_figure(path_makefig)
    break
#############################################################################
