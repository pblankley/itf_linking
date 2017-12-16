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
from libr import kepcart as kc
import collections
import astropy
from clustering import member_counts
from collections import defaultdict
from collections import Counter
from libr import MPC_library
import scipy.spatial
import pickle
from operator import add
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cleaning as cl

def fit_tracklet_grav(t_ref, g, gdot, v, GM=MPC_library.Constants.GMsun, eps2=1e-16):
    """ NOT USED YET. JUST HERE FOR POTENTIAL FUTURE USE.
    Here's a more sophisticated version. """

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

def allocate_with_fits(final_dict,indiv_fits,method='fit'):
    """ This function allocates tracklets to clusters. Since with a KD Tree it is
    possible to have clusters that contain the same tracklet, this step is necessary.
    NOTE: this function is meant primairly for internal use in the cluster of months.
    ----------
    Args: final_dict; dict, a dictionary keyed on clusters that are represented with
                the usual '|' joined strings of tracklet ids
    ----------
    Returns: a dict keyed on tracklet id and the new (potentially smaller) cluster
                id as the value.
       nf_dict docs
    """
    cld, nf_dict, all_trkls = {}, {}, set()
    valid_methods = ['fit','smallest','largest','coin']

    # Determine method of allocation.
    if method not in valid_methods:
        raise ValueError('Enter a valid method or go with the default')
    if method=='fit':
        all_clusts = [i[0] for i in sorted([(k,v[1]) for k,v in final_dict.items()],key=lambda x: x[1])]
    elif method=='smallest':
        all_clusts = sorted(final_dict.keys(),key=lambda x: len(x))
    elif method=='largest':
        all_clusts = sorted(final_dict.keys(),key=lambda x: len(x),reverse=True)
    elif method=='coin':
        all_clusts = list(final_dict.keys())

    # Allocate
    for i, str_cid in enumerate(all_clusts):
        trkls = set(str_cid.split('|'))
        all_trkls |= trkls
        trkls = sorted(list(trkls))
        tv = [tid in cld.keys() for tid in trkls]
        if not any(tv):
            for tid in trkls:
                cld[tid] = '|'.join(trkls)
                temp_f+=indiv_fits[str_cid][tid][0]
                temp_e+=indiv_fits[str_cid][tid][1]
            nf_dict['|'.join(trkls)] = (temp_f/len(trkls),temp_e/len(trkls))
        else:
            if (len(tv)-sum(tv))>=3:
                nids = [tid for tr,tid in zip(tv,trkls) if not tr]
                for tid in nids:
                    cld[tid] = '|'.join(nids)
                    temp_f+=indiv_fits[str_cid][tid][0]
                    temp_e+=indiv_fits[str_cid][tid][1]
                nf_dict['|'.join(nids)] = (temp_f/len(nids),temp_e/len(nids))
    # Deal with the null (unassigned) tracklets.
    null_trkls = [tid for tid in all_trkls if tid not in cld.keys()]
    for nt in null_trkls:
        cld[nt] = 'abstain'

    return nf_dict, cld

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

def gsr_files(infilenames, pixel, nside, n, angDeg=5.5, g=0.4, gdot=0.0, cluster_id_dict={}):
    """ This function was supposed to help create a meta cluster plot, but the healpix vector
    format was too difficult to manipulate in the time we had, so this file is in misc.py"""
    hp_dicts = []
    for infn in infilenames:
        hp_dicts.append(util.get_hpdict(infn))

    vec = hp.pix2vec(nside, pixel, nest=True)
    neighbors = hp.query_disc(nside, vec, angDeg*np.pi/180., inclusive=True, nest=True)
    lines = []
    for pix in neighbors:
        for hp_dict in hp_dicts:
            for line in hp_dict[pix]:
                lines.append(line)

    outfilename = 'demo_train/training_quiver_meta'
    if len(lines) > 0:
        transform_to_arrows(util.lunation_center(n), [(g, gdot)], vec, lines, outfilename, makefiles=True, cluster_id_dict=cluster_id_dict)


def cluster_months_copy(fit_dicts, rad, alloc=False, alloc_method='fit', GM=MPC_library.Constants.GMsun):
    """ This function will take in the fit_dict output of the postprocessing step
    and find then cluster the orbital elements (or parameters) based on a reference
    time.  This will hopefully product valid, month to month clusters.
    ---------     all the fit_dicts for every month
    Args: fit_dicts where the key is string cluster_id and the value is the a tuple with
                EITHER the related, fitted a,adot,b,bdot,g,gdot parameters for that
                cluster OR the realted, transformed a, e, i, big_omega, little_omega, m
                from the orbital elements transform, and array of observation level errors.
          times where times is a list of the same length as fit dicts and contains the julian
                data associated with the fit_dict at the related index.
          t_ref, float, the reference time we are scaling everything back to. get this
                value by calling util.lunation_center()
    ---------
    Returns: fit_dict where the key is string cluster_id and the value is the a tuple with
                EITHER the related, fitted a,adot,b,bdot,g,gdot parameters for that
                cluster OR the realted, transformed a, e, i, big_omega, little_omega, m
                from the orbital elements transform, and array of observation level errors.
                The difference here is that the fit dict has been clustered and re-fit.
    ---------
    Note: n = np.sqrt(GM/a**3); m = m+n*(t-t_ref) for the time transformation.
          We are not doing this transformation because it only has to do with the
          position of the asteroid, not the orbit itself.
    """
    final_dict,indiv_orbs = {},{}
    points,labels = [],[]
    for fit_dict in fit_dicts:
        for clust, orb in fit_dict.items():
            # a, e, i, bo, lo, m = orb[0]
            labels.append((clust,np.mean(orb[1])))
            points.append(orb[0][:5])

    tree = scipy.spatial.cKDTree(np.array(points))
    matches = tree.query_ball_tree(tree, rad)

    for i,match in enumerate(matches):
        if len(match)>0:
            cluster_list =[]
            orb_params,orb_errors = [points[i]],[labels[i][1]]
            for idx in match:
                tracklet_ids = labels[idx][0].split('|')
                cluster_list.extend(tracklet_ids)
                orb_params.append(points[idx])
                orb_errors.append(labels[idx][1])
            cluster_key='|'.join(sorted(cluster_list))

            # A very significant amount of housekeeping
            # trkl_fits = [f for f,cl in sorted(zip(orb_params,cluster_list),key=lambda x: x[1])]
            # trkl_errs = [e for e,cl in sorted(zip(orb_errors,cluster_list),key=lambda x: x[1])]
            final_dict[cluster_key] = (np.array(orb_params).mean(axis=0), np.mean(orb_errors))
            # indiv_orbs[cluster_key] = {}
            # for i,tid in enumerate(sorted(cluster_list)):
            #     if tid not in indiv_orbs[cluster_key].keys():
            #         print('cl length',cluster_list)
            #         print('i',i,'tid',tid)
            #         print('length f',trkl_fits)
            #         print('legnth e',trkl_errs)
            #         indiv_orbs[cluster_key][tid] = [(trkl_fits[i], trkl_errs[i])]
            #     else:
            #         indiv_orbs[cluster_key][tid].append((trkl_fits[i], trkl_errs[i]))
            # for tid,v in indiv_orbs[cluster_key].items():
            #     indiv_orbs[cluster_key][tid] = (sum(i[0] for i in v)/len(v), sum(i[1] for i in v)/len(v))


    if alloc:
        final_dict, cid_dict = util.allocate_with_fits(final_dict,indiv_orbs,method=alloc_method)
    else:
        cid_dict = get_cid_dict(final_dict, shared=False)
    return final_dict, cid_dict


def clean_demo_data():
    # This routine is making a smaller ITF for demonstration purposes.  It has just three months of data.
    #
    def grab_time_range(infilename='data/itf_new_1_line.txt', outfilename='data/itf_new_1_line_subset.txt',
                        lower_time=2457263.5, upper_time=2457353.5):
        with open(outfilename, 'w') as outfile:
            with open(infilename, 'r') as f:
                for line in f:
                    objName, provDesig, disAst, note1, note2, dateObs, RA, Dec, mag, filt, obsCode = cl.convertObs80(line)
                    jd_utc = MPC_library.date2JD(dateObs)
                    jd_tdb  = MPC_library.EOP.jdTDB(jd_utc)
                    if jd_tdb>=lower_time and jd_tdb<upper_time:
                        outfile.write(line)


    grab_time_range()

    grab_time_range(infilename='data/UnnObs_Training_1_line_A.txt', outfilename='data/UnnObs_Training_1_line_A_subset.txt')

    # This is inactivated because the results have already been generated and don't need to be redone.
    #
    with open('itf_new_1_line_subset.mpc', 'w') as outfile:
        with open('itf_new_1_line_subset.txt', 'r') as f:
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
                if filt.isspace():
                    filt = '-'
                if mag.isspace():
                    mag = '----'
                xh, yh, zh = Observatories.getObservatoryPosition(obsCode, jd_utc)
                outstring = "%11s %s %4s %5s %s %13.6lf %12.7lf %12.7lf %12.7lf %12.6lf %12.7lf %12.7lf\n"% \
                      (provDesig, dateObs, obsCode, mag, filt, jd_tdb, x, y, z, xh, yh, zh)
                outfile.write(outstring)

    tracklets, tracklets_jd_dict, sortedTracklets = cl.get_sorted_tracklets('forPaul/itf_new_1_line_subset.mpc')

    for k in sortedTracklets[:10]:
        print(k, tracklets_jd_dict[k])

    len(sortedTracklets)

    UnnObs_tracklets, UnnObs_tracklets_jd_dict, UnnObs_sortedTracklets = cl.get_sorted_tracklets('forPaul/UnnObs_Training_1_line_A_subset.mpc')

    for k in UnnObs_sortedTracklets[:10]:
        print(k, UnnObs_tracklets_jd_dict[k])

    len(UnnObs_sortedTracklets)

    cl.separate_time_windows(tracklets, sortedTracklets, tracklets_jd_dict, file_stem='forPaul/itf_new_1_line_subset.mpc', dt=15.)
    cl.separate_time_windows(UnnObs_tracklets, UnnObs_sortedTracklets, UnnObs_tracklets_jd_dict, file_stem='forPaul/UnnObs_Training_1_line_A_subset.mpc', dt=15.)
