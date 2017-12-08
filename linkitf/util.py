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

Observatories = MPC_library.Observatories

ObservatoryXYZ = Observatories.ObservatoryXYZ

def lunation_center(n, tref=2457722.0125, p=29.53055):
    """ Returns the jd of new moon, to the nearest half day"""
    t = tref + p*n
    tp = np.floor(t) + 0.5
    return tp

def equatorial_to_ecliptic(v,rot_mat=MPC_library.rotate_matrix(-MPC_library.Constants.ecl)):
    """ convert equatorial plane x,y,z to eliptic x,y,z """
    return np.dot(rot_mat,v.reshape(-1,1)).flatten()

def xyz_to_proj_matrix(r_ref):
    """ This routine returns the 3-D rotation matrix for the
    given reference vector."""
    x_ref, y_ref, z_ref = r_ref
    r = np.sqrt(x_ref**2 + y_ref**2 + z_ref**2)
    lon0 = np.arctan2(y_ref, x_ref)
    lat0 = np.arcsin(z_ref/r)
    slon0 = np.sin(lon0)
    clon0 = np.cos(lon0)
    slat0 = np.sin(lat0)
    clat0 = np.cos(lat0)

    mat = np.array([[-slon0, clon0, 0],
                    [-clon0*slat0, -slon0*slat0, clat0],
                    [clon0*clat0, slon0*clat0, slat0 ]])
    return mat

def make_histogram(mergedCounter):
    """ helper function for processing of merged counter dict from
    accessible_clusters function"""
    hist_dict=defaultdict(list)
    for k, v in mergedCounter.items():
        hist_dict[v].append(k)
    return hist_dict

def number_of_occurances(mergedCounter_dict, nside):
    """ counts the number of occurances of clusters of each size"""
    num_occ=Counter()
    for pix in range(hp.nside2npix(nside)):
        hist=make_histogram(mergedCounter_dict[pix])
        for k, v in hist.items():
            num_occ.update({k: len(v)})
    return num_occ

def get_hpdict(infilename):
    """ Takes in a .trans file and returns the related healpix dictionary.
    We get the healpix number from the last column of the file and make a
    key, value pair where the healpix number of the key and the line is the
    value.
    -------
    Args: str, filename to the .trans file
    -------
    Returns: dict
    """
    hp_dict = defaultdict(list)
    with open(infilename) as file:
        for line in file:
            if line.startswith('#'):
                continue
            pix = int(line.split()[-1])
            hp_dict[pix].append(line)
    return hp_dict


def get_original_tracklets_dict(filename):
    """ orig default was ='data/UnnObs_Training_1_line_A.mpc'"""
    tracklets = defaultdict(list)
    with open(filename) as infile:
        for i, line in enumerate(infile):
            if not line.startswith('#'):
                desig = line[0:12].strip()
                tracklets[desig].append(i)
    return tracklets


def get_original_observation_array(filename):
    """ orig default was ='data/UnnObs_Training_1_line_A.txt' """
    tracklets = defaultdict(list)
    with open(filename) as infile:
        data = infile.readlines()
    return data

def get_observations(cluster_key, tracklets_dict, observation_array, sep='|'):
    """ 2nd and 3rd args are from the above functions"""
    array=[]
    for key in cluster_key.split(sep):
        indices = tracklets_dict[key]
        for idx in indices:
            array.append(observation_array[idx].rstrip())
    return array












############ for later TODO

def fit_tracklet_grav(t_ref, g, gdot, v, GM=MPC_library.Constants.GMsun, eps2=1e-16):
    """ Here's a more sophisticated version. """

    t_emit = [(obs[0]-obs[1]-t_ref) for obs in v]

    # We will approximate g_x(t), g_y(t), and g_z(t)
    # using a Taylor expansion.
    # The first two terms are zero by design.
    #
    # Given alpha, beta, gamma,
    # we would have r_0^2 = (alpha^2 + beta^2 + 1)*z_0^2
    # r^2 = (alpha^2 + beta^2 + 1)/gamma^2 ~ 1/gamma^2
    # g_x(t) ~ -0.5*GM*x_0*t^2/r^3 + 1/6*jerk_x*t*t*t
    # g_y(t) ~ -0.5*GM*y_0*t^2/r^3 + 1/6*jerk_y*t*t*t
    # g_z(t) ~ -0.5*GM*z_0*t^2/r^3 + 1/6*jerk_z*t*t*t
    #
    # We do not have alpha and beta initially,
    # but we assert gamma.
    #
    # We set alpha=beta=0 initially, least squares solve
    # the tracklets and obtain alpha, alpha-dot, beta,
    # and beta-dot.
    #
    # Then we use those values to estimate g_x,
    # g_y, and g_z for the next iteration.
    #
    # The process converges when alpha, alpha-dot,
    # beta, beta-dot do not change significantly.
    #
    # We could also do this same process with a
    # Kepler-stepper or a full n-body integration.

    alpha = beta = 0.0
    alpha_dot = beta_dot = 0.0
    cx, cy = 1.0, 1.0
    mx, my = 0.0, 0.0

    while(((cx-alpha)*(cx-alpha) + (cy-beta)*(cy-beta))>eps2):

        alpha, beta = cx, cy
        alpha_dot, beta_dot = mx, my

        r2 = (alpha*alpha + beta*beta + 1.0)/(g*g)
        r3 = r2*np.sqrt(r2)
        r5 = r2*r3

        x0 = alpha/g
        y0 = beta/g
        z0 = 1.0/g

        vx0 = alpha_dot/g
        vy0 = beta_dot/g
        vz0 = gdot/g

        # An alternative to the Taylor expansion is to
        # to kepler step from
        # x0, y0, z0 and vx0, vy0, vz0 at time 0
        # to those at the times of each observation
        # in the tracklet.  With that there will be no
        # issue of convergence.
        # Then simply subtract off the linear motion
        # to give the gravitational perturbation.

        rrdot = x0*vx0 + y0*vy0 + z0*vz0

        acc_x = -GM*x0/r3
        acc_y = -GM*y0/r3
        acc_z = -GM*z0/r3

        jerk_x = -GM/r5*(r2*vx0 - 3.0*rrdot*x0)
        jerk_y = -GM/r5*(r2*vy0 - 3.0*rrdot*y0)
        jerk_z = -GM/r5*(r2*vz0 - 3.0*rrdot*z0)

        fac =[(1.0 + gdot*t + 0.5*g*acc_z*t*t + (1./6.0)*g*jerk_z*t*t*t - g*obs[7]) for obs, t in zip(v, t_emit)]

        A = np.vstack([t_emit, np.ones(len(t_emit))]).T

        x = [obs[2]*f + obs[5]*g - 0.5*g*acc_x*t*t - (1./6.0)*g*jerk_x*t*t*t for obs, f, t in zip(v, fac, t_emit)]
        mx, cx = np.linalg.lstsq(A, x)[0]

        y = [obs[3]*f + obs[6]*g - 0.5*g*acc_y*t*t - (1./6.0)*g*jerk_y*t*t*t for obs, f, t in zip(v, fac, t_emit)]
        my, cy = np.linalg.lstsq(A, y)[0]

        return (cx, mx, cy, my, t_emit[0])

# # Select positions and cluster positions compare TODO
    # def select_positions_z(t_ref, g, gdot, vec, lines, outfilename, fit_tracklet_func=fit_tracklet):
    #     """
    #     This is the one to use.  This routine will be used repeatedly.
    #
    #     Trying a slightly different approach.
    #     The set of lines that are being passed in have
    #     been selected to be in the same region of sky
    #     for an assumed distance.  We are going to re-transform
    #     those assuming a fixed z (or gamma) value with respect
    #     to the sun and the reference direction, rather than a
    #     fixed r, at the reference time
    #
    #     Rotate observatory positions to projection coordinates,
    #     and recalculate simple z-based light-time correction.
    #
    #     Rotate the observations to projection coordinates,
    #     but they will be theta_x, theta_y only
    #
    #     Fit the simple abg model, for fixed gamma and
    #     possibly gamma_dot.
    #     """
    #     #GM = MPC_library.Constants.GMsun
    #
    #     # This rotation is taking things from equatorial to ecliptic
    #     # rot_mat = MPC_library.rotate_matrix(-MPC_library.Constants.ecl)
    #
    #     results_dict = defaultdict(list)
    #
    #     # vec is the reference direction in eliptic coordinates
    #     vec = np.array(vec)
    #     # ecl_vec = np.dot(rot_mat, vec)
    #     # vec = ecl_vec
    #     vec = vec/np.linalg.norm(vec)
    #     # mat is a rotation matrix that converts from ecliptic
    #     # vectors to the projection coordinate system.
    #     # The projection coordinate system has z outward,
    #     # x parallel to increasing ecliptic longitude, and
    #     # y northward, making a right-handed system.
    #     mat = xyz_to_proj_matrix(vec)
    #
    #     # Loop over all the lines from a *.trans file.
    #     for line in lines:
    #         if line.startswith('#'):
    #             # Concatenate all header lines?
    #             header = line.rstrip()
    #         else:
    #             lineID = line[:43]
    #             trackletID = line[0:12]
    #
    #             jd_tdb = float(line[43:57])
    #             dtp = float(line[139:150])
    #
    #             # Get unit vector to target
    #             x_target, y_target, z_target = line[58:98].split()
    #             r_target = np.array([float(x_target), float(y_target), float(z_target)])
    #
    #             # Rotate to ecliptic coordinates
    #             # r_target_ec = np.dot(rot_mat, r_target)
    #
    #             # Rotate to projection coordinates
    #             theta_x, theta_y, theta_z = np.dot(mat, r_target)
    #
    #             # Ignore theta_z after this; it should be very nearly 1.
    #
    #             # Get observatory position, ultimately in projection coordinates.
    #             x_obs, y_obs, z_obs = line[98:138].split()
    #             r_obs = np.array([float(x_obs), float(y_obs), float(z_obs)])
    #
    #             # Rotate to ecliptic coordinates
    #             # r_obs_ec = np.dot(rot_mat, r_obs)
    #
    #             # Rotate to projection coordinates
    #             xe, ye, ze = np.dot(mat, r_obs)
    #
    #             # This is the light travel time
    #             dlt = ze/MPC_library.Constants.speed_of_light
    #
    #             # Append the resulting data to a dictionary keye do trackletID.
    #             results_dict[trackletID].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))
    #
    #     # Now that we have the observations for each tracklet gathered together,
    #     # we iterate through the tracklets, doing a fit for each one.
    #     results = []
    #     for k, v in results_dict.items():
    #
    #         cx, mx, cy, my, t0 = fit_tracklet_func(t_ref, g, gdot, v)
    #         outstring = "%12s %16.9lf %16.9lf %16.9lf %16.9lf %16.9lf\n" % (k, cx, mx, cy, my, t0)
    #
    #         '''
    #
    #         # Here's a version that incorporates radial gravitational
    #         # acceleration
    #
    #         t_emit = [(obs[0]-obs[1]-t_ref) for obs in v]
    #         acc_z = -GM*g*g
    #         fac =[(1.0 + gdot*t + 0.5*g*acc_z*t*t - g*obs[7]) for obs, t in zip(v, t_emit)]
    #
    #         A = np.vstack([t_emit, np.ones(len(t_emit))]).T
    #
    #         # Can I put a simple acc_x term in here?
    #         x = [obs[2]*f + obs[5]*g for obs, f in zip(v, fac)]
    #         mx, cx = np.linalg.lstsq(A, x)[0]
    #
    #         y = [obs[3]*f + obs[6]*g for obs, f in zip(v, fac)]
    #         my, cy = np.linalg.lstsq(A, y)[0]
    #
    #         outstring = "%12s %16.9lf %16.9lf %16.9lf %16.9lf %16.9lf\n" % (k, cx, mx, cy, my, t_emit[0])
    #         '''
    #
    #         results.append(outstring)
    #
    #     if len(results)>0:
    #         with open(outfilename, 'w') as outfile:
    #             outstring = '#  g = %lf\n' % (g)
    #             outfile.write(outstring)
    #             outstring = '#  gdot = %lf\n' % (gdot)
    #             outfile.write(outstring)
    #             outstring = '#  vec= %lf, %lf, %lf\n' % (vec[0], vec[1], vec[2])
    #             outfile.write(outstring)
    #             outstring = '#  desig              alpha         alpha_dot       beta             beta_dot         dt \n'
    #             outfile.write(outstring)
    #             for outstring in results:
    #                 outfile.write(outstring)
    #
    #
    # def cluster_positions_z(t_ref, g_gdot_pairs, vec, lines, fit_tracklet_func=fit_tracklet):
    #     """ Cluster function that gets passed to the cluster_sky_regions function
    #     Here I am doing the same thing as the previous routine, but without files.
    #
    #     It takes a reference time (t_ref), a set of z, zdot pairs (z_zdot_pairs),
    #     a reference direction vector (vec), and a set of observation lines that
    #     have been selected for a region of sky and time slice (lines)
    #
    #     It returns a dictionary of results that have z, zdot pairs as keys and
    #     sets of fitted tracklets as results.  Each result has the form:
    #
    #     trackletID alpha alpha_dot beta beta_dot t_emit,
    #     where t_emit is the light time-corrected time relative to the reference
    #     time.  The coordinates are now in tangent plane projection.
    #
    #     """
    #     #GM = MPC_library.Constants.GMsun
    #
    #     # rot_mat = MPC_library.rotate_matrix(-MPC_library.Constants.ecl)
    #
    #     results_dict = defaultdict(list)
    #
    #     vec = np.array(vec)
    #     # ecl_vec = np.dot(rot_mat, vec)
    #     # vec = ecl_vec
    #     vec = vec/np.linalg.norm(vec)
    #     mat = xyz_to_proj_matrix(vec)
    #
    #     for line in lines:
    #         if line.startswith('#'):
    #             header = line.rstrip()
    #         else:
    #             lineID = line[:43]
    #             trackletID = line[0:12]
    #
    #             jd_tdb = float(line[43:57])
    #             dtp = float(line[139:150])
    #
    #             # Get unit vector to target
    #             x_target, y_target, z_target = line[58:98].split()
    #             r_target = np.array([float(x_target), float(y_target), float(z_target)])
    #
    #             # Rotate to ecliptic coordinates
    #             # r_target_ec = np.dot(rot_mat, r_target)
    #
    #             # Rotate to projection coordinates
    #             theta_x, theta_y, theta_z = np.dot(mat, r_target)
    #
    #             # Ignore theta_z after this; it should be very nearly 1.
    #
    #             # Get observatory position
    #             x_obs, y_obs, z_obs = line[98:138].split()
    #             r_obs = np.array([float(x_obs), float(y_obs), float(z_obs)])
    #
    #             # Rotate to ecliptic coordinates
    #             # r_obs_ec = np.dot(rot_mat, r_obs)
    #
    #             # Rotate to projection coordinates
    #             xe, ye, ze = np.dot(mat, r_obs)
    #
    #             dlt = ze/MPC_library.Constants.speed_of_light
    #
    #             results_dict[trackletID].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))
    #
    #     # All the work done above is independent of the z0 and zdot0 values
    #
    #     master_results = {}
    #     for g_gdot in g_gdot_pairs:
    #         g, gdot = g_gdot
    #
    #         results = []
    #         for k, v in results_dict.items():
    #
    #             cx, mx, cy, my, t0 = fit_tracklet_func(t_ref, g, gdot, v)
    #             result = (k, cx, mx, cy, my, t0)
    #             results.append(result)
    #
    #         master_results[g_gdot] = results
    #
    #     return master_results
# def do_training_run(pixels, infilename, t_ref,
#                     cluster_sky_function=cluster_sky_regions,
#                     g_gdots=g_gdots, mincount=3,
#                     dts=np.arange(5, 30, 5),
#                     radii=np.arange(0.0001, 0.0100, 0.0001)):
#
#     master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)
#
#     results_dict = {}
#
#     rates_dict={}
#     for dt in dts:
#         for rad in radii:
#             cluster_counter = Counter()
#             for pix, d in master.items():
#                 for g_gdot, arrows in d.items():
#
#                     # The bit from here
#                     i = 0
#                     label_dict={}
#                     combined=[]
#                     for k, cx, mx, cy, my, t in arrows:
#                         label_dict[i] = k
#                         combined.append([cx, mx*dt, cy, my*dt])
#                         i +=1
#                     points=np.array(combined)
#                     # to here can be a function,
#                     # that takes arrows and dt and
#                     # returns label_dict and points array
#
#                     # The bit from here
#                     tree = scipy.spatial.cKDTree(points)
#                     matches = tree.query_ball_tree(tree, rad)
#                     # to here can be a function, that takes
#                     # points are rad and returns tree and
#                     # matches
#
#                     for j, match in enumerate(matches):
#                         if len(match)>=mincount:
#                             cluster_list =[]
#                             tracklet_params=[]
#                             for idx in match:
#                                 cluster_list.append(label_dict[idx].strip())
#                                 #tracklet_params.append(combined[idx])
#                             cluster_key='|'.join(sorted(cluster_list))
#                             cluster_counter.update({cluster_key: 1})
#
#             # This region from here
#             errs = 0
#             for i, k in enumerate(cluster_counter.keys()):
#                 keys = k.split('|')
#                 stems = [key.split('_')[0] for key in keys]
#                 stem_counter = Counter(stems)
#                 if len(stem_counter)>1:
#                     errs +=1
#             # to here can be a function that takes a concatenated
#             # cluster ID and returns the number of errors
#
#             rates_dict[dt, rad] = cluster_counter.keys(), errs
#
#     for dt in dts:
#         values = []
#         for k, v in rates_dict.items():
#             dtp, d = k
#             if dtp==dt:
#                 test_set = list(v[0])
#                 ncs, nes = len(unique_clusters(test_set)[0]), len(unique_clusters(test_set)[1])
#                 values.append((d, ncs, nes, test_set))
#
#         values = sorted(values, key=lambda v: v[0])
#         ds = [v[0] for v in values]
#         nclusters = [v[1] for v in values]
#         nerrors = [v[2] for v in values]
#         keys = [v[3] for v in values]
#         results_dict[dt] = ds, nclusters, nerrors, keys
#
#     return results_dict
#
# def do_test_run(pixels, infilename, t_ref,
#                     cluster_sky_function=cluster_sky_regions,
#                     g_gdots=g_gdots, mincount=3,
#                     dt=15,
#                     rad=0.00124):
#
#     master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)
#
#     results_dict={}
#     rates_dict={}
#     cluster_counter = Counter()
#     for pix, d in master.items():
#         for z_zdot, arrows in d.items():
#
#             # The bit from here
#             i = 0
#             label_dict={}
#             combined=[]
#             for k, cx, mx, cy, my, t in arrows:
#                 label_dict[i] = k
#                 combined.append([cx, mx*dt, cy, my*dt])
#                 i +=1
#             points=np.array(combined)
#             # to here can be a function,
#             # that takes arrows and dt and
#             # returns label_dict and points array
#
#             # The bit from here
#             tree = scipy.spatial.cKDTree(points)
#             matches = tree.query_ball_tree(tree, rad)
#             # to here can be a function, that takes
#             # points are rad and returns tree and
#             # matches
#
#             for j, match in enumerate(matches):
#                 if len(match)>=mincount:
#                     # From here is about making a cluster key
#                     # from the indices and label dictionary
#                     cluster_list =[]
#                     for idx in match:
#                         cluster_list.append(label_dict[idx].strip())
#                         #tracklet_params.append(combined[idx])
#                     cluster_key='|'.join(sorted(cluster_list))
#                     # to here
#
#                     cluster_counter.update({cluster_key: 1})
#
#     test_set = list(cluster_counter.keys())
#     success_dict, failure_counter = unique_clusters(test_set)
#
#     values = len(success_dict), len(failure_counter), list(success_dict.keys()), list(failure_counter.keys())
#
#     return values
#
#
# gs = [0.3, 0.35, 0.4, 0.45, 0.5]
# gs =[0.4]
# gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
# g_gdots = [(x,y) for x in gs for y in gdots]
#
# def do_run(pixels, infilename, t_ref,
#                     cluster_sky_function=cluster_sky_regions,
#                     g_gdots=g_gdots, mincount=3,
#                     dt=15,
#                     rad=0.00124):
#
#     master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)
#
#     results_dict={}
#     rates_dict={}
#     cluster_counter = Counter()
#     for pix, d in master.items():
#         for z_zdot, arrows in d.items():
#
#             # The bit from here
#             i = 0
#             label_dict={}
#             combined=[]
#             for k, cx, mx, cy, my, t in arrows:
#                 label_dict[i] = k
#                 combined.append([cx, mx*dt, cy, my*dt])
#                 i +=1
#             points=np.array(combined)
#             # to here can be a function,
#             # that takes arrows and dt and
#             # returns label_dict and points array
#
#             # The bit from here
#             tree = scipy.spatial.cKDTree(points)
#             matches = tree.query_ball_tree(tree, rad)
#             # to here can be a function, that takes
#             # points are rad and returns tree and
#             # matches
#
#             for j, match in enumerate(matches):
#                 if len(match)>=mincount:
#                     # From here is about making a cluster key
#                     # from the indices and label dictionary
#                     cluster_list =[]
#                     for idx in match:
#                         cluster_list.append(label_dict[idx].strip())
#                         #tracklet_params.append(combined[idx])
#                     cluster_key='|'.join(sorted(cluster_list))
#                     # to here
#
#                     cluster_counter.update({cluster_key: 1})
#
#     return cluster_counter

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
    mat = xyz_to_proj_matrix(vec)

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
