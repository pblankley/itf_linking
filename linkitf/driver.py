# Imports
import numpy as np
import scipy.interpolate
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import math
import lib.kepcart as kc
import healpy as hp
import collections
import astropy
from collections import defaultdict
from collections import Counter
from lib import MPC_library
import scipy.spatial
import pickle
from operator import add
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import util
import cleaning as cl
from clustering import find_clusters
from itf_clean import clean_itf_data

# clean_itf_data('data/here_is_itf.txt')
with open('demo_itf/itf_clust_ids.pickle', 'rb') as handle:
    itf_clust_ids = pickle.load(handle)
print(type(itf_clust_ids))
print(list(itf_clust_ids.keys())[:10])
print(list(itf_clust_ids.values())[:10])
