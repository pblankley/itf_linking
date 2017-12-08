# Imports
import numpy as np
import scipy.interpolate
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import math
# import lib.kepcart as kc
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
from clustering import find_clusters
from itf_clean import clean_itf_data_mpc
import os


print('enter the path where you have the large mpc file. (no quotes)')
mpcpath = input()
mpc_path = os.path.abspath(mpcpath)
txtpath = input()
txt_path = os.path.abspath(txtpath)
clean_itf_data_mpc(mpc_path)

gs = [0.4]
gdots = [-0.004, -0.002, 0.0, 0.002, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]
nside=8
dt=15.0
cr=0.00124
pixels = range(hp.nside2npix(nside))


for n in range(-825,14):
    itf_file = os.path.join(home_dir, 'itf_new_1_line_ec_{}_pm15.0_r2.5.trans'.format(str(util.lunation_center(n))))
    itf_raw_results, itf_clust_ids = find_clusters(pixels, itf_file, util.lunation_center(n), g_gdots=g_gdots,dt=dt,rad=cr)

    itf_tracklets_dict = util.get_original_tracklets_dict(os.path.join(mpc_path))
    itf_obs_array = util.get_original_observation_array(os.path.join(txt_path))

    obs_dict={}
    for cluster_key in itf_raw_results.keys():
        obs_dict[cluster_key] = util.get_observations(cluster_key, itf_tracklets_dict, itf_obs_array)

    with open('result_for_{}.pickle'.format(n),'wb') as handle:
        pickle.dump(obs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        











# with open('demo_itf/itf_clust_ids.pickle', 'rb') as handle:
#     itf_clust_ids = pickle.load(handle)
# print(type(itf_clust_ids))
# print(list(itf_clust_ids.keys())[:10])
# print(list(itf_clust_ids.values())[:10])
