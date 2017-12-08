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
from lib import MPC_library
import scipy.spatial
import pickle
from operator import add
import cleaning as cl
import util
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# os.path.join(BASE_DIR, 'data/plist_df.json')

def clean_itf_data(path):
    itf_path = os.path.abspath(path)
    home_dir = os.path.dirname(itf_path)

    cl.split_MPC_file(itf_path)

    _txt_to_mpc_after_split(itf_path, home_dir)

    mpc_path = os.path.join(home_dir,'itf_new_1_line_ec.mpc')

    tracklets, tracklets_jd_dict, sortedTracklets = cl.get_sorted_tracklets(mpc_path)


    cl.separate_time_windows(tracklets, sortedTracklets, tracklets_jd_dict, \
                            file_stem=mpc_path, dt=15.)

    cl.separate_time_windows(tracklets, sortedTracklets, tracklets_jd_dict, \
                            file_stem=itf_path, dt=15., suff='.txt')

    for n in range(-825,14):
        cl.index_positions(n, lambda t: 2.5, file_stem=mpc_path, dt=15.)

    return


def _txt_to_mpc_after_split(full_path,home_dir):
    """ helper function for above clean itf data"""
    Observatories = MPC_library.Observatories

    with open(os.path.join(home_dir,'itf_new_1_line_ec.mpc'), 'w') as outfile:
        with open(full_path, 'r') as f:
            outstring = "#trackletID yr   mn dy      obsCode mag filter  jd_tdb       x_target     y_target     z_target      x_obs       y_obs        z_obs     \n"
            outfile.write(outstring)
            for line in f:
                objName, provDesig, disAst, note1, note2, dateObs, RA, Dec, mag, filt, obsCode = cl.convertObs80(line)
                jd_utc = MPC_library.date2JD(dateObs)
                jd_tdb  = MPC_library.EOP.jdTDB(jd_utc)
                raDeg, decDeg = MPC_library.RA2degRA(RA), MPC_library.Dec2degDec(Dec)
                x = np.cos(decDeg*np.pi/180.)*np.cos(raDeg*np.pi/180.)
                y = np.cos(decDeg*np.pi/180.)*np.sin(raDeg*np.pi/180.)
                z = np.sin(decDeg*np.pi/180.)
                xec, yec, zec = util.equatorial_to_ecliptic(np.array((x, y, z)))

                if filt.isspace():
                    filt = '-'
                if mag.isspace():
                    mag = '----'
                xh, yh, zh = Observatories.getObservatoryPosition(obsCode, jd_utc)
                xhec, yhec, zhec = util.equatorial_to_ecliptic(np.array((xh, yh, zh)))
                #outstring = "%11s %s %4s %5s %s %13.6lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf\n"% \
                #      (provDesig, dateObs, obsCode, mag, filt, jd_tdb, x, y, z, xh, yh, zh, xec, yec, zec, xhec, yhec, zhec)
                outstring = "%11s %s %4s %5s %s %13.6lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf\n"% \
                    (provDesig, dateObs, obsCode, mag, filt, jd_tdb, xec, yec, zec, xhec, yhec, zhec)
                outfile.write(outstring)


def clean_training_data(path):
    train_path = os.path.abspath(path)
    home_dir = os.path.dirname(train_path)

    cl.split_MPC_file(train_path)

    _txt_to_mpc_after_split(train_path, home_dir)

    mpc_path = os.join.path(home_dir,'UnnObs_Training_1_line_A_ec.mpc')

    UnnObs_tracklets, UnnObs_tracklets_jd_dict, UnnObs_sortedTracklets = cl.get_sorted_tracklets(mpc_path)

    cl.separate_time_windows(UnnObs_tracklets, UnnObs_sortedTracklets, UnnObs_tracklets_jd_dict, \
                        file_stem=mpc_path, dt=15.)

    cl.separate_time_windows(UnnObs_tracklets, UnnObs_sortedTracklets, UnnObs_tracklets_jd_dict, \
                            file_stem=mpc_path, dt=45.)
    for n in range(-825,14):
        cl.index_positions(n, lambda t: 2.5, file_stem=mpc_path, dt=15.)

    for n in range(-825,14):
        cl.index_positions(n, lambda t: 2.5, file_stem=mpc_path, dt=45.)


def  _txt_to_mpc_after_split(full_path, home_dir):
    """ Helper function for the above clean_training_data"""
    Observatories = MPC_library.Observatories

    with open(os.join.path(home_dir,'UnnObs_Training_1_line_A_ec.mpc'), 'w') as outfile:
        with open(full_path, 'r') as f:
            outstring = "#trackletID yr   mn dy      obsCode mag filter  jd_tdb       x_target     y_target     z_target      x_obs       y_obs        z_obs     \n"
            outfile.write(outstring)
            for line in f:
                objName, provDesig, disAst, note1, note2, dateObs, RA, Dec, mag, filt, obsCode = cl.convertObs80(line)
                jd_utc = MPC_library.date2JD(dateObs)
                jd_tdb  = MPC_library.EOP.jdTDB(jd_utc)
                raDeg, decDeg = MPC_library.RA2degRA(RA), MPC_library.Dec2degDec(Dec)
                x = np.cos(decDeg*np.pi/180.)*np.cos(raDeg*np.pi/180.)
                y = np.cos(decDeg*np.pi/180.)*np.sin(raDeg*np.pi/180.)
                z = np.sin(decDeg*np.pi/180.)
                xec, yec, zec = util.equatorial_to_ecliptic(np.array((x, y, z)))

                if filt.isspace():
                    filt = '-'
                if mag.isspace():
                    mag = '----'
                xh, yh, zh = Observatories.getObservatoryPosition(obsCode, jd_utc)
                xhec, yhec, zhec = util.equatorial_to_ecliptic(np.array((xh, yh, zh)))
                #outstring = "%11s %s %4s %5s %s %13.6lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf\n"% \
                #      (provDesig, dateObs, obsCode, mag, filt, jd_tdb, x, y, z, xh, yh, zh, xec, yec, zec, xhec, yhec, zhec)
                outstring = "%11s %s %4s %5s %s %13.6lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf\n"% \
                    (provDesig, dateObs, obsCode, mag, filt, jd_tdb, xec, yec, zec, xhec, yhec, zhec)
                outfile.write(outstring)
