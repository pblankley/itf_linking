# Imports
import numpy as np
import scipy.interpolate
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import math
import kepcart as kc
import healpy as hp
import collections
import astropy
from collections import defaultdict
from collections import Counter
import MPC_library
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


def xyz_to_proj_matrix(r_ref):
    """ This routine returns the 3-D rotation matrix for the
    given reference vector."""
    x_ref, y_ref, z_ref = r_ref
    r = np.sqrt(x_ref*x_ref + y_ref*y_ref + z_ref*z_ref)
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


def select_positions_z(t_ref, g, gdot, vec, lines, outfilename, fit_tracklet_func=fit_tracklet):
    """
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
    """
    #GM = MPC_library.Constants.GMsun

    # This rotation is taking things from equatorial to ecliptic
    rot_mat = MPC_library.rotate_matrix(MPC_library.Constants.ecl)

    results_dict = defaultdict(list)

    # vec is the reference direction in equatorial coordinates
    # so we rotate to ecliptic, because we want to.
    vec = np.array(vec)
    ecl_vec = np.dot(rot_mat, vec)
    vec = ecl_vec
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
            r_target_ec = np.dot(rot_mat, r_target)

            # Rotate to projection coordinates
            theta_x, theta_y, theta_z = np.dot(mat, r_target_ec)

            # Ignore theta_z after this; it should be very nearly 1.

            # Get observatory position, ultimately in projection coordinates.
            x_obs, y_obs, z_obs = line[98:138].split()
            r_obs = np.array([float(x_obs), float(y_obs), float(z_obs)])

            # Rotate to ecliptic coordinates
            r_obs_ec = np.dot(rot_mat, r_obs)

            # Rotate to projection coordinates
            xe, ye, ze = np.dot(mat, r_obs_ec)

            # This is the light travel time
            dlt = ze/MPC_library.Constants.speed_of_light

            # Append the resulting data to a dictionary keye do trackletID.
            results_dict[trackletID].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))

    # Now that we have the observations for each tracklet gathered together,
    # we iterate through the tracklets, doing a fit for each one.
    results = []
    for k, v in results_dict.items():

        cx, mx, cy, my, t0 = fit_tracklet_func(t_ref, g, gdot, v)
        outstring = "%12s %16.9lf %16.9lf %16.9lf %16.9lf %16.9lf\n" % (k, cx, mx, cy, my, t0)

        '''

        # Here's a version that incorporates radial gravitational
        # acceleration

        t_emit = [(obs[0]-obs[1]-t_ref) for obs in v]
        acc_z = -GM*g*g
        fac =[(1.0 + gdot*t + 0.5*g*acc_z*t*t - g*obs[7]) for obs, t in zip(v, t_emit)]

        A = np.vstack([t_emit, np.ones(len(t_emit))]).T

        # Can I put a simple acc_x term in here?
        x = [obs[2]*f + obs[5]*g for obs, f in zip(v, fac)]
        mx, cx = np.linalg.lstsq(A, x)[0]

        y = [obs[3]*f + obs[6]*g for obs, f in zip(v, fac)]
        my, cy = np.linalg.lstsq(A, y)[0]

        outstring = "%12s %16.9lf %16.9lf %16.9lf %16.9lf %16.9lf\n" % (k, cx, mx, cy, my, t_emit[0])
        '''

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


def transform_to_arrows(t_ref, g_gdot_pairs, vec, lines, fit_tracklet_func=fit_tracklet):
    """ Cluster function that gets passed to the cluster_sky_regions function
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
    #GM = MPC_library.Constants.GMsun

    rot_mat = MPC_library.rotate_matrix(MPC_library.Constants.ecl)

    results_dict = defaultdict(list)

    vec = np.array(vec)
    ecl_vec = np.dot(rot_mat, vec)
    vec = ecl_vec
    vec = vec/np.linalg.norm(vec)
    mat = xyz_to_proj_matrix(vec)

    for line in lines:
        if line.startswith('#'):
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
            r_target_ec = np.dot(rot_mat, r_target)

            # Rotate to projection coordinates
            theta_x, theta_y, theta_z = np.dot(mat, r_target_ec)

            # Ignore theta_z after this; it should be very nearly 1.

            # Get observatory position
            x_obs, y_obs, z_obs = line[98:138].split()
            r_obs = np.array([float(x_obs), float(y_obs), float(z_obs)])

            # Rotate to ecliptic coordinates
            r_obs_ec = np.dot(rot_mat, r_obs)

            # Rotate to projection coordinates
            xe, ye, ze = np.dot(mat, r_obs_ec)

            dlt = ze/MPC_library.Constants.speed_of_light

            results_dict[trackletID].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))

    # All the work done above is independent of the z0 and zdot0 values

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
        # print(i)
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



# gs = [0.3, 0.35, 0.4, 0.45, 0.5]
gs =[0.4]
gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

# This could accept a cluster_sky regions function, so that there is only one version of this
# code.

# And it should accept dt and rad ranges.

### TODO: incorporate this training decision maker:
"""
error_rate_limit=1e-3
training_dict={}
for n in [-11, -14, -17, -20, -23]:
    infilename='data/UnnObs_Training_1_line_A_%.1lf_pm15.0_r2.5.trans' % (lunation_center(n))
    pickle_filename = infilename.rstrip('trans') + 'v2_pickle'
    dt=15
    with open(pickle_filename, 'rb') as handle:
        pix_runs = pickle.load(handle)

        # Should save these results in files
        true_count_dict, mergedCounter_dict, mergedTime_dict=accessible_clusters(list(pix_runs.keys()), infilename=infilename)
        true_count=sum(true_count_dict.values())
        true_count

        print(n)
        for i in range(99):
            errs=0
            clusts=0
            trues=0
            for pix in list(pix_runs.keys()):
                nclusters = pix_runs[pixels[pix]][dt][1][i]
                nerrors = pix_runs[pixels[pix]][dt][2][i]
                ntrue = true_count_dict[pix]

                errs += nerrors
                clusts += nclusters
                trues += ntrue
            if float(errs)/trues<error_rate_limit:
                print(i, pix_runs[pixels[pix]][dt][0][i], errs, clusts, trues)
            else:
                training_dict[n] = pix_runs[pixels[pix]][dt][0][i], errs, clusts, trues
                break
"""

def do_training_run(pixels, infilename, t_ref,
                    cluster_sky_function=cluster_sky_regions,
                    g_gdots=g_gdots, mincount=3,
                    dts=np.arange(5, 30, 5),
                    radii=np.arange(0.0001, 0.0100, 0.0001)):

    master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)

    results_dict = {}

    rates_dict={}
    for dt in dts:
        for rad in radii:
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
                            tracklet_params=[]
                            for idx in match:
                                cluster_list.append(label_dict[idx].strip())
                                #tracklet_params.append(combined[idx])
                            cluster_key='|'.join(sorted(cluster_list))
                            cluster_counter.update({cluster_key: 1})

            # This region from here
            errs = 0
            for i, k in enumerate(cluster_counter.keys()):
                keys = k.split('|')
                stems = [key.split('_')[0] for key in keys]
                stem_counter = Counter(stems)
                if len(stem_counter)>1:
                    errs +=1
            # to here can be a function that takes a concatenated
            # cluster ID and returns the number of errors

            rates_dict[dt, rad] = cluster_counter.keys(), errs

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

def make_histogram(mergedCounter):
    hist_dict=defaultdict(list)
    for k, v in mergedCounter.items():
        hist_dict[v].append(k)
    return hist_dict


def number_clusters_plot(pix_runs,true_count):
    for dt in np.arange(5, 30, 5):
        pixels=list(pix_runs.keys())
        ds = pix_runs[pixels[0]][dt][0]
        nclusters = pix_runs[pixels[0]][dt][1]
        nerrors = pix_runs[pixels[0]][dt][2]
        for pix in pixels[1:]:
            nclusters = list(map(add, nclusters, pix_runs[pix][dt][1]))
            nerrors = list(map(add, nerrors, pix_runs[pix][dt][2]))
        nclusters=np.array(nclusters)

        plt.plot(ds, nclusters, label=dt)

    plt.axhline(true_count, ls='dashed')
    plt.xlabel('d (cluster radius)')
    plt.ylabel('N clusters')
    #plt.text(0.005, 400, r'$\gamma=0.4$', fontsize=15)
    plt.legend()
    plt.title('Number of clusters by cluster radius.')
    plt.savefig('demo_data/nclusters_demo')
    plt.show()


def number_errors_plot(pix_runs):
    for dt in np.arange(5, 30, 5):
        pixels=list(pix_runs.keys())
        ds = pix_runs[pixels[0]][dt][0]
        nclusters = pix_runs[pixels[0]][dt][1]
        nerrors = pix_runs[pixels[0]][dt][2]
        for pix in pixels[1:]:
            nclusters = list(map(add, nclusters, pix_runs[pix][dt][1]))
            nerrors = list(map(add, nerrors, pix_runs[pix][dt][2]))
        nclusters=np.array(nclusters)
        nerrors=np.array(nerrors)
        '''
        if dt==50:
            for d, nc, ne in zip(ds, nclusters, nerrors):
                print(d, nc, ne)
        '''


        plt.plot(ds, nerrors, label=dt)

    plt.xscale("log", nonposx='clip')
    plt.ylim((0,3000))
    plt.xlabel('d (cluster radius)')
    plt.ylabel('N errors')
    plt.text(0.0005, 1000, r'$\gamma=0.4$', fontsize=15)
    plt.legend()
    plt.title('Number of errors by cluster radius')
    plt.savefig('demo_data/nerrors_demo')
    plt.show()


def auc_plot(pix_runs,true_count):
    for dt in np.arange(5, 30, 5):
        pixels=list(pix_runs.keys())
        ds = pix_runs[pixels[0]][dt][0]
        nclusters = pix_runs[pixels[0]][dt][1]
        nerrors = pix_runs[pixels[0]][dt][2]
        for pix in pixels[1:]:
            nclusters = list(map(add, nclusters, pix_runs[pix][dt][1]))
            nerrors = list(map(add, nerrors, pix_runs[pix][dt][2]))
        nclusters=np.array(nclusters)
        nerrors=np.array(nerrors)

        plt.plot(nerrors/true_count, nclusters/true_count, label=dt)

    plt.xlim((0, 0.02))
    plt.ylim((0, 1))
    plt.xlabel('Error rate')
    plt.ylabel('Fraction complete')
    plt.text(0.05, 0.2, r'$\gamma=0.4$', fontsize=15)
    #plt.legend()
    plt.title('AUC-proxy plot')
    plt.savefig('demo_data/AUC_demo')
    plt.show()


gs = [0.3, 0.35, 0.4, 0.45, 0.5]
gs =[0.4]
gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

def do_test_run(pixels, infilename, t_ref,
                    cluster_sky_function=cluster_sky_regions,
                    g_gdots=g_gdots, mincount=3,
                    dt=15,
                    rad=0.00124):

    master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)

    results_dict={}
    rates_dict={}
    cluster_counter = Counter()
    for pix, d in master.items():
        for z_zdot, arrows in d.items():

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
                    # From here is about making a cluster key
                    # from the indices and label dictionary
                    cluster_list =[]
                    for idx in match:
                        cluster_list.append(label_dict[idx].strip())
                        #tracklet_params.append(combined[idx])
                    cluster_key='|'.join(sorted(cluster_list))
                    # to here

                    cluster_counter.update({cluster_key: 1})

    test_set = list(cluster_counter.keys())
    success_dict, failure_counter = unique_clusters(test_set)

    values = len(success_dict), len(failure_counter), list(success_dict.keys()), list(failure_counter.keys())

    return values

""" look at above method and change codee in demo accordingly
def evaluate(pix_runs, true_count_dict, earth_vec, dt=15, nside=8):
    errs=0
    clusts=0
    trues=0
    for pix in list(pix_runs.keys()):
        pixels=list(pix_runs.keys())
        nclusters = pix_runs[pixels[pix]][dt][1][4]
        nerrors = pix_runs[pixels[pix]][dt][2][4]

        ntrue = true_count_dict[pix]
        vec = hp.pix2vec(nside, pix, nest=True)
        ang=180./np.pi * np.arccos(np.dot(earth_vec/np.linalg.norm(vec), vec))
        if ang<180:
            #print("%5d %5d %5d %5d %7.2lf" % (pix, nclusters, nerrors, ntrue, ang))
            errs += nerrors
            clusts += nclusters
            trues += ntrue
    return errs, clusts, trues
"""

gs = [0.3, 0.35, 0.4, 0.45, 0.5]
gs =[0.4]
gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

def do_run(pixels, infilename, t_ref,
                    cluster_sky_function=cluster_sky_regions,
                    g_gdots=g_gdots, mincount=3,
                    dt=15,
                    rad=0.00124):

    master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)

    results_dict={}
    rates_dict={}
    cluster_counter = Counter()
    for pix, d in master.items():
        for z_zdot, arrows in d.items():

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
                    # From here is about making a cluster key
                    # from the indices and label dictionary
                    cluster_list =[]
                    for idx in match:
                        cluster_list.append(label_dict[idx].strip())
                        #tracklet_params.append(combined[idx])
                    cluster_key='|'.join(sorted(cluster_list))
                    # to here

                    cluster_counter.update({cluster_key: 1})

    return cluster_counter

# Should just passing the selection function, to avoid code duplication
def generate_sky_region_files(infilename, nside=8, n=-11, angDeg=5.5, g=0.4, gdot=0.0):
    hp_dict = defaultdict(list)
    with open(infilename) as file:
        for line in file:
            if line.startswith('#'):
                continue
            pix = int(line.split()[-1])
            hp_dict[pix].append(line)

    for i in range(hp.nside2npix(nside)):
        vec = hp.pix2vec(nside, i, nest=True)
        neighbors = hp.query_disc(32, vec, angDeg*np.pi/180., inclusive=True, nest=True)
        lines = []
        for pix in neighbors:
            for line in hp_dict[pix]:
                lines.append(line)
        outfilename = infilename.rstrip('.trans') + '_hp_' + ('%03d' % (i)) + '_g'+ ('%.2lf' % (g))+'_gdot' + ('%+5.1le' % (gdot))
        if len(lines) > 0:
            select_positions_z(lunation_center(n), g, gdot, vec, lines, outfilename)


def make_figure(filename):
    plt.ioff()
    mxs, cxs, mys, cys, dts =[], [], [], [], []
    for line in open(filename):
        if line.startswith('#'):
            continue
        desig, cx, mx, cy, my, dt = line.split()
        mxs.append(float(mx))
        cxs.append(float(cx))
        mys.append(float(my))
        cys.append(float(cy))
        dts.append(float(dt))

    fig=plt.figure(figsize=(18, 16))

    #norm = Normalize()
    #norm.autoscale(colors)

    colormap = cm.inferno

    plt.quiver(cxs, cys, mxs, mys, dts, scale=0.3, width=0.0003)

    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    outfile = filename+'.pdf'
    plt.savefig(outfile)
    plt.close()
    plt.ion()
