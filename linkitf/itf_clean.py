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

def clean_itf_data(path):

    Observatories = MPC_library.Observatories

    cl.split_MPC_file(path)

    # This is inactivated because the results have already been generated and don't need to be redone.
    #
    # This is inactivated because the results have already been generated and don't need to be redone.
    #
    with open('data/itf_new_1_line_ec.mpc', 'w') as outfile:
        with open('data/itf_new_1_line.txt', 'r') as f:
            outstring = "#trackletID yr   mn dy      obsCode mag filter  jd_tdb       x_target     y_target     z_target      x_obs       y_obs        z_obs     \n"
            outfile.write(outstring)
            for line in f:
                objName, provDesig, disAst, note1, note2, dateObs, RA, Dec, mag, filt, obsCode = convertObs80(line)
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


    tracklets, tracklets_jd_dict, sortedTracklets = cl.get_sorted_tracklets('data/itf_new_1_line.mpc')

    for k in sortedTracklets[:10]:
        print(k, tracklets_jd_dict[k])

    print('We have {} observations for ITF tracklets.'.format(len(sortedTracklets)))

    UnnObs_tracklets, UnnObs_tracklets_jd_dict, UnnObs_sortedTracklets = cl.get_sorted_tracklets('data/UnnObs_Training_1_line_A.mpc')

    for k in UnnObs_sortedTracklets[:10]:
        print(k, UnnObs_tracklets_jd_dict[k])

    print('We have {} observations for training data.'.format(len(UnnObs_sortedTracklets)))

    cl.separate_time_windows(tracklets, sortedTracklets, tracklets_jd_dict, \
                            file_stem='data/itf_new_1_line.mpc', dt=15.)

    cl.separate_time_windows(UnnObs_tracklets, UnnObs_sortedTracklets, UnnObs_tracklets_jd_dict, \
                            file_stem='data/UnnObs_Training_1_line_A.mpc', dt=15.)

    cl.separate_time_windows(tracklets, sortedTracklets, tracklets_jd_dict, \
                            file_stem='data/itf_new_1_line.txt', dt=15., suff='.txt')

    cl.separate_time_windows(UnnObs_tracklets, UnnObs_sortedTracklets, UnnObs_tracklets_jd_dict, \
                            file_stem='data/UnnObs_Training_1_line_A.mpc', dt=45.)

    for n in range(-825,14):
        cl.index_positions(n, lambda t: 2.5, file_stem='data/itf_new_1_line.mpc', dt=15.)

    for n in range(-825,14):
        cl.index_positions(n, lambda t: 2.5, file_stem='data/UnnObs_Training_1_line_A.mpc', dt=15.)

    for n in range(-825,14):
        cl.index_positions(n, lambda t: 2.5, file_stem='data/UnnObs_Training_1_line_A.mpc', dt=45.)

def clean_training_data():
    # This is inactivated because the results have already been generated and don't need to be redone.
    #
    with open('data/UnnObs_Training_1_line_A_ec.mpc', 'w') as outfile:
        with open('data/UnnObs_Training_1_line_A.txt', 'r') as f:
            outstring = "#trackletID yr   mn dy      obsCode mag filter  jd_tdb       x_target     y_target     z_target      x_obs       y_obs        z_obs     \n"
            outfile.write(outstring)
            for line in f:
                objName, provDesig, disAst, note1, note2, dateObs, RA, Dec, mag, filt, obsCode = convertObs80(line)
                jd_utc = MPC_library.date2JD(dateObs)
                jd_tdb  = MPC_library.EOP.jdTDB(jd_utc)
                raDeg, decDeg = MPC_library.RA2degRA(RA), MPC_library.Dec2degDec(Dec)
                x = np.cos(decDeg*np.pi/180.)*np.cos(raDeg*np.pi/180.)
                y = np.cos(decDeg*np.pi/180.)*np.sin(raDeg*np.pi/180.)
                z = np.sin(decDeg*np.pi/180.)
                xec, yec, zec = equatorial_to_ecliptic(np.array((x, y, z)))

                if filt.isspace():
                    filt = '-'
                if mag.isspace():
                    mag = '----'
                xh, yh, zh = Observatories.getObservatoryPosition(obsCode, jd_utc)
                xhec, yhec, zhec = equatorial_to_ecliptic(np.array((xh, yh, zh)))
                #outstring = "%11s %s %4s %5s %s %13.6lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf\n"% \
                #      (provDesig, dateObs, obsCode, mag, filt, jd_tdb, x, y, z, xh, yh, zh, xec, yec, zec, xhec, yhec, zhec)
                outstring = "%11s %s %4s %5s %s %13.6lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf\n"% \
                    (provDesig, dateObs, obsCode, mag, filt, jd_tdb, xec, yec, zec, xhec, yhec, zhec)
                outfile.write(outstring)

    
