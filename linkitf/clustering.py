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

Observatories = MPC_library.Observatories

ObservatoryXYZ = Observatories.ObservatoryXYZ

def fit_tracklet(t_ref, g, gdot, v, GM=MPC_library.Constants.GMsun):
    """Here's a version that incorporates radial gravitational
    acceleration. """

    t_emit = [(obs[0]-obs[1]-t_ref) for obs in v]
    acc_z = -GM*g*g
    fac =[(1.0 + gdot*t + 0.5*g*acc_z*t*t - g*obs[7]) for obs, t in zip(v, t_emit)]

    A = np.vstack([t_emit, np.ones(len(t_emit))]).T

    x = [obs[2]*f + obs[5]*g for obs, f in zip(v, fac)]
    mx, cx = np.linalg.lstsq(A, x)[0]

    y = [obs[3]*f + obs[6]*g for obs, f in zip(v, fac)]
    my, cy = np.linalg.lstsq(A, y)[0]

    return (cx, mx, cy, my, t_emit[0])


def get_tracklet_obs(vec,lines):
    results_dict = defaultdict(list)

    # vec is the reference direction in equatorial coordinates
    # so we rotate to ecliptic, because we want to.
    vec = np.array(vec)
    # ecl_vec = np.dot(rot_mat, vec)
    # vec = ecl_vec
    vec = vec/np.linalg.norm(vec)
    # mat is a rotation matrix that converts from ecliptic
    # vectors to the projection coordinate system.
    # The projection coordinate system has z outward,
    # x parallel to increasing ecliptic longitude, and
    # y northward, making a right-handed system.
    mat = util.xyz_to_proj_matrix(vec)

    # Loop over all the lines from a *.trans file.
    for line in lines:
        if line.startswith('#'):
            # Concatenate all header lines?
            header = line.rstrip()
        else:
            lineID = line[:43]
            trackletID = line[0:12]

            jd_tdb = float(line[43:57])
            dtp = float(line[139:150])

            # Get unit vector to target
            x_target, y_target, z_target = line[58:98].split()
            r_target = np.array([float(x_target), float(y_target), float(z_target)])

            # Rotate to ecliptic coordinates
            # r_target_ec = np.dot(rot_mat, r_target)

            # Rotate to projection coordinates
            theta_x, theta_y, theta_z = np.dot(mat, r_target)

            # Ignore theta_z after this; it should be very nearly 1.

            # Get observatory position, ultimately in projection coordinates.
            x_obs, y_obs, z_obs = line[98:138].split()
            r_obs = np.array([float(x_obs), float(y_obs), float(z_obs)])

            # Rotate to ecliptic coordinates
            # r_obs_ec = np.dot(rot_mat, r_obs)

            # Rotate to projection coordinates
            xe, ye, ze = np.dot(mat, r_obs)

            # This is the light travel time
            dlt = ze/MPC_library.Constants.speed_of_light

            # Append the resulting data to a dictionary keye do trackletID.
            results_dict[trackletID].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))

    return results_dict
# cluster positions z inputs: t_ref, g_gdot_pairs, vec, lines, fit_tracklet_func=fit_tracklet
def _return_arrows_resuts(results_dict, t_ref, g_gdot_pairs, fit_tracklet_func):
        # Now that we have the observations for each tracklet gathered together,
        # we iterate through the tracklets, doing a fit for each one.
    master_results = {}
    for g_gdot in g_gdot_pairs:
        g, gdot = g_gdot

        results = []
        for k, v in results_dict.items():

            cx, mx, cy, my, t0 = fit_tracklet_func(t_ref, g, gdot, v)
            result = (k, cx, mx, cy, my, t0)
            results.append(result)

        master_results[g_gdot] = results
    return master_results

def _write_arrows_files(results_dict, t_ref, g_gdot_pairs, fit_tracklet_func, vec, outfilename):
        # Now that we have the observations for each tracklet gathered together,
        # we iterate through the tracklets, doing a fit for each one.
    for g_gdot in g_gdot_pairs:
        g, gdot = g_gdot
        results = []
        for k, v in results_dict.items():

            cx, mx, cy, my, t0 = fit_tracklet_func(t_ref, g, gdot, v)
            outstring = "%12s %16.9lf %16.9lf %16.9lf %16.9lf %16.9lf\n" % (k, cx, mx, cy, my, t0)

            results.append(outstring)

        if len(results)>0:
            with open(outfilename, 'w') as outfile:
                outstring = '#  g = %lf\n' % (g)
                outfile.write(outstring)
                outstring = '#  gdot = %lf\n' % (gdot)
                outfile.write(outstring)
                outstring = '#  vec= %lf, %lf, %lf\n' % (vec[0], vec[1], vec[2])
                outfile.write(outstring)
                outstring = '#  desig              alpha         alpha_dot       beta             beta_dot         dt \n'
                outfile.write(outstring)
                for outstring in results:
                    outfile.write(outstring)

def transform_to_arrows(t_ref, g_gdot_pairs, vec, lines, outfilename='', makefiles=False, fit_tracklet_func=fit_tracklet):
    """
    Args: g_gdot pairs is a list of tuples with the g, gdot pairs to use
            for the select_clusters_z functionality, pass the g, gdot in
            [(g,gdot)] format
    This is the one to use.  This routine will be used repeatedly.

    Trying a slightly different approach.
    The set of lines that are being passed in have
    been selected to be in the same region of sky
    for an assumed distance.  We are going to re-transform
    those assuming a fixed z (or gamma) value with respect
    to the sun and the reference direction, rather than a
    fixed r, at the reference time

    Rotate observatory positions to projection coordinates,
    and recalculate simple z-based light-time correction.

    Rotate the observations to projection coordinates,
    but they will be theta_x, theta_y only

    Fit the simple abg model, for fixed gamma and
    possibly gamma_dot.
    Cluster function that gets passed to the cluster_sky_regions function
    Here I am doing the same thing as the previous routine, but without files.

    It takes a reference time (t_ref), a set of z, zdot pairs (z_zdot_pairs),
    a reference direction vector (vec), and a set of observation lines that
    have been selected for a region of sky and time slice (lines)

    It returns a dictionary of results that have z, zdot pairs as keys and
    sets of fitted tracklets as results.  Each result has the form:

    trackletID alpha alpha_dot beta beta_dot t_emit,
    where t_emit is the light time-corrected time relative to the reference
    time.  The coordinates are now in tangent plane projection.
    """
    results_dict = get_tracklet_obs(vec,lines)

    if not makefiles:
        return _return_arrows_resuts(results_dict, t_ref, g_gdot_pairs, fit_tracklet_func)
    else:
        _write_arrows_files(results_dict, t_ref, g_gdot_pairs, fit_tracklet_func, vec, outfilename)

def cluster_sky_regions(g_gdot_pairs, pixels, t_ref, infilename, nside=8, angDeg=7.5, cluster_func=transform_to_arrows):
    """ cluster function that gets passed to the do_training_run"""
    # This bit from here
    hp_dict = defaultdict(list)
    with open(infilename) as file:
        for line in file:
            if line.startswith('#'):
                continue
            pix = int(line.split()[-1])
            hp_dict[pix].append(line)
    # to here can be a function that
    # accepts infilename and returns
    # hp_dict

    pixel_results = {}
    #for i in range(hp.nside2npix(nside)):
    for i in pixels:
        print(i)
        # Probably don't need to repeat the vector neighbor calculation.
        # This can just be stored.
        vec = hp.pix2vec(nside, i, nest=True)
        neighbors = hp.query_disc(nside, vec, angDeg*np.pi/180., inclusive=True, nest=True)
        lines = []
        for pix in neighbors:
            for line in hp_dict[pix]:
                lines.append(line)
        if len(lines) > 0:
            pixel_results[i] = cluster_func(t_ref, g_gdot_pairs, vec, lines)

    return pixel_results

gs =[0.4]
gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

def _get_cluster_counter(master, dt, rad, mincount):
    cluster_counter = Counter()
    for pix, d in master.items():
        for g_gdot, arrows in d.items():

            # The bit from here
            i = 0
            label_dict={}
            combined=[]
            for k, cx, mx, cy, my, t in arrows:
                label_dict[i] = k
                combined.append([cx, mx*dt, cy, my*dt])
                i +=1
            points=np.array(combined)
            # to here can be a function,
            # that takes arrows and dt and
            # returns label_dict and points array

            # The bit from here
            tree = scipy.spatial.cKDTree(points)
            matches = tree.query_ball_tree(tree, rad)
            # to here can be a function, that takes
            # points are rad and returns tree and
            # matches

            for j, match in enumerate(matches):
                if len(match)>=mincount:
                    cluster_list =[]
                    for idx in match:
                        cluster_list.append(label_dict[idx].strip())
                    cluster_key='|'.join(sorted(cluster_list))
                    cluster_counter.update({cluster_key: 1})
    return cluster_counter

def _rates_to_results(rates_dict, dts):
    """ helper for find clusters"""
    for dt in dts:
        values = []
        for k, v in rates_dict.items():
            dtp, d = k
            if dtp==dt:
                test_set = list(v[0])
                ncs, nes = len(unique_clusters(test_set)[0]), len(unique_clusters(test_set)[1])
                values.append((d, ncs, nes, test_set))

        values = sorted(values, key=lambda v: v[0])
        ds = [v[0] for v in values]
        nclusters = [v[1] for v in values]
        nerrors = [v[2] for v in values]
        keys = [v[3] for v in values]
        results_dict[dt] = ds, nclusters, nerrors, keys

    return results_dict

def find_clusters(pixels, infilename, t_ref, g_gdots=g_gdots,
                    rtype='run', dt=15, rad=0.00124,
                    cluster_sky_function=cluster_sky_regions, mincount=3):
    """docs"""

    valid_rtypes = ['run','test','train']

    if rtype not in valid_rtypes:
        raise ValueError('You must use a rtype in {0}, not {1}'.format(valid_rtypes,rtype))

    if (hasattr(dt, '__len__') or hasattr(rad, '__len__')) and rtype!='train':
        raise ValueError('test and run rtypes can only take scalar dt and rad values')

    if not hasattr(rad, '__len__') and rtype=='train':
        raise ValueError('train rtypes can only take a list or array of dt and rad values')

    master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)

    if rtype=='run':
        return _get_cluster_counter(master, dt, rad, mincount)

    elif rtype=='test':
        cluster_counter = _get_cluster_counter(master, dt, rad, mincount)
        test_set = list(cluster_counter.keys())
        success_dict, failure_counter = unique_clusters(test_set)
        return len(success_dict), len(failure_counter), list(success_dict.keys()), list(failure_counter.keys())
    else:
        # The training case
        results_dict = {}
        rates_dict={}
        for dt_val in dt:
            for rad_val in rad:
                cluster_counter = _get_cluster_counter(master, dt_val, rad_val, mincount)
                # This region from here
                errs = 0
                for i, k in enumerate(cluster_counter.keys()):
                    keys = k.split('|')
                    stems = [key.split('_')[0] for key in keys]
                    stem_counter = Counter(stems)
                    if len(stem_counter)>1:
                        errs +=1

                rates_dict[dt_val, rad_val] = cluster_counter.keys(), errs

        return _rates_to_results(rates_dict, dt)




## TODO why is the n not specified and alwasy -11???
def output_sky_regions(pixels, infilename, nside=8, n=-11, angDeg=7.5):
    hp_dict = defaultdict(list)
    with open(infilename) as file:
        i=0
        for line in file:
            if line.startswith('#'):
                continue
            pix = int(line.split()[-1])
            hp_dict[pix].append(line)
            i += 1

    pixel_results = {}
    for i in pixels:
        # Probably don't need to repeat the vector neighbor calculation.
        # This can just be stored.
        vec = hp.pix2vec(nside, i, nest=True)
        neighbors = hp.query_disc(nside, vec, angDeg*np.pi/180., inclusive=True, nest=True)
        lines = []
        for pix in neighbors:
            for line in hp_dict[pix]:
                lines.append(line)

    return lines


def accessible_clusters(pixels, infilename, mincount=3):
    true_counts={}
    mergedCounter_dict = {}
    mergedTime_dict = {}
    for pix in pixels:
        lines = output_sky_regions([pix], infilename=infilename)
        #print(pix, len(lines))
        trackletCounter = Counter()
        tracklet_time = defaultdict(float)
        for line in lines:
            trackletID=line.split()[0]
            trackletCounter.update({trackletID : 1})
            jd_tdb = float(line[43:57])
            tracklet_time[trackletID] = jd_tdb

        mergedCounter = Counter()
        time_dict = defaultdict(list)
        for k, v in trackletCounter.items():
            mergedCounter.update({k[:-4]:1})
            time_dict[k[:-4]].append(tracklet_time[k])
        true_counts[pix]=len([k for k, v in mergedCounter.items() if v>=mincount])
        mergedCounter_dict[pix]=mergedCounter
        mergedTime_dict[pix]=time_dict

    return true_counts, mergedCounter_dict, mergedTime_dict

def member_counts(k, sep='|', suff='_'):
    """ helper function for unique clusters, counts the instances of an id in a cluster """

    keys = k.split(sep)
    stems = [key.split(suff)[0] for key in keys]
    stem_counter = Counter(stems)
    return stem_counter

def unique_clusters(test_set):
    """ clustering based on the prefix of the results id_003, id_002.. etc"""
    success_dict = {}
    failure_counter = Counter()
    for k in test_set:
        stem_counter = member_counts(k)
        if len(stem_counter)>1:
            failure_counter.update({k:1})
        else:
            for stem, v in stem_counter.items():
                if stem not in success_dict:
                    success_dict[stem] = v, k
                elif v > success_dict[stem][0]:
                    success_dict[stem] = v, k
    return success_dict, failure_counter

# Should just passing the selection function, to avoid code duplication
def generate_sky_region_files(infilename, nside, n, angDeg=5.5, g=0.4, gdot=0.0):
    hp_dict = defaultdict(list)
    with open(infilename) as file:
        for line in file:
            if line.startswith('#'):
                continue
            pix = int(line.split()[-1])
            hp_dict[pix].append(line)

    for i in range(hp.nside2npix(nside)):
        vec = hp.pix2vec(nside, i, nest=True)
        neighbors = hp.query_disc(nside, vec, angDeg*np.pi/180., inclusive=True, nest=True)
        lines = []
        for pix in neighbors:
            for line in hp_dict[pix]:
                lines.append(line)
        outfilename = infilename.rstrip('.trans') + '_hp_' + ('%03d' % (i)) + '_g'+ ('%.2lf' % (g))+'_gdot' + ('%+5.1le' % (gdot))
        if len(lines) > 0:
            transform_to_arrows(lunation_center(n), [(g, gdot)], vec, lines, outfilename, makefiles=True)
