# Imports
import numpy as np
import scipy.interpolate
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
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
import matplotlib.colors as mlc
import matplotlib.cm as cm
import util
from copy import copy


Observatories = MPC_library.Observatories

ObservatoryXYZ = Observatories.ObservatoryXYZ

def fit_tracklet(t_ref, g, gdot, v, GM=MPC_library.Constants.GMsun):
    """ This function used least squares to fit the a, adot and the b, bdot in
    two seperate equations.  The equations are seperate and uncoupled because we
    chose the gamma value (g) to be the value we set, and let gdot follow.
    This follows the mathematical scheme from Bernstein et al (2000).
    This version also incorporates radial gravitational acceleration.
    ----------
    Args: t_ref; the lunation center
          g; float, the gamma value we assert.
          gdot; float, the gamma dot value we assert.
          v; the list of tuples, where each tuple is an observation and the
                meta-list is a tracklet.
    ----------
    Returns: tuple, with (a, adot, b, bdot and t_emit) as the values.
                These values are on the tracklet level.
    """
    t_emit = [(obs[0]-obs[1]-t_ref) for obs in v]
    acc_z = -GM*g*g
    fac =[(1.0 + gdot*t + 0.5*g*acc_z*t*t - g*obs[7]) for obs, t in zip(v, t_emit)]

    A = np.vstack([t_emit, np.ones(len(t_emit))]).T

    x = [obs[2]*f + obs[5]*g for obs, f in zip(v, fac)]
    mx, cx = np.linalg.lstsq(A, x)[0]

    y = [obs[3]*f + obs[6]*g for obs, f in zip(v, fac)]
    my, cy = np.linalg.lstsq(A, y)[0]
        # a     adot b  bdot  dt (change in time)
    return (cx, mx, cy, my, t_emit[0])

def error(opt_result,args,flag='chi'):
    """ Calculate the error for the result of the minimization function.
    returns the mean chi sq or rmse and the associated chi sq or rmse array
    on the observation level.  The function is meant to test for the validity of
    clusters fit with our nonlinear curve fitting algorithm.
    ---------
    Args: opt_result, array length 6, the result of the optimization function.
          args: the list of lists that contains each individual observation in
                    the cluster and its xe, ye, ze, t_emit, theta_x, and theta_y.
          flag: str, either 'chi', or 'rms'. Designates which error term is calculated.
    ---------
    Returns: tuple of form (float, list) where the float is the rmse or avg chi sq
                and the list, which has length equal to the number of observations,
                is the rmse or chi sq on a per-observation level.
    """
    vflag = ('chi','rms')
    if flag not in vflag:
        raise ValueError('flag not in valid flags try rms or chi')
    if flag=='chi':
        err,var_e=0.0,(0.2/205625.)**2 # in radians (from arcseconds)
    if flag=='rms':
        err,var_e=0.0,(1./205625.)**2

    a,adot,b,bdot,g,gdot = opt_result
    err_arr = []
    for arg in args:
        xe, ye, ze, t_emit, theta_x, theta_y = arg
        tx = (a + adot*t_emit - g*xe)/(1 + gdot*t_emit - g*ze)
        ty = (b + bdot*t_emit - g*ye)/(1 + gdot*t_emit - g*ze)
        if flag=='rms':
            calc_err = np.sqrt((theta_x-tx)**2/var_e + (theta_y-ty)**2/var_e)
        elif flag=='chi':
            calc_err = (theta_x-tx)**2/var_e + (theta_y-ty)**2/var_e
        err += calc_err
        err_arr.append(calc_err)

    return err/len(err_arr),err_arr

def full_fit_t_loss(t_ref, g_init, gdot_init,  list_of_tracklets, flag='rms', GM=MPC_library.Constants.GMsun,
                    tol=None, use_jacobian=True, method='BFGS', details=False):
    """ This function needs to take in all the observations over the cluster of
    tracklets (min of 3 tracklets), and return the a,adot,b,bdot,g and gdot.

    We will then use the resulting gamma and gamma dot to fit the tracklets
    individually, and compare with the chi sq.
    --------
    Args: t_ref; lunation_center
          g_init; out initial guess for g (the value we asserted before)
          gdot_init; out initial guess for gdot (the value we asserted before)
          all_obs; list of tuples, where each tuple is a observation and all the
                    observations make up at minimum 3 tracklets in a cluster.
          tol; float, the maximum tolerance we have for error in our minimization
          use_jacobian; bool, a T/F flag for whether to use the jacobian or not.
          method; str, specifies the method used to minimize the loss. Must be one
                    of 'Nelder-Mead', 'COBYLA','L-BFGS-B','Powell','BFGS','TNC','dogleg',
                    'trust-ncg','SLSQP','Newton-CG', or 'CG'.
          details; bool, flag for the user to specify if they want the extra
                    information about the minimization (number of iterations
                    of the solver).
    --------
    Returns: tuple with (params, function min, chisq) where the first element is
                parameters calculated by the nonlinear fit, the second is the
                value of the loss function when completely minimized, the third
                value is the chisq statistic, and the fourth is the array of
                every observation in the cluster and its related error term.

            NOTE: if details=True then this output has the features in the details
                    description added to it.
    """
    valid_methods = ['Nelder-Mead','COBYLA','L-BFGS-B','Powell','BFGS','TNC','dogleg','trust-ncg','SLSQP','Newton-CG','CG']
    nosupport_jacobian = ['Nelder-Mead','COBYLA','Powell']
    if method not in valid_methods:
        raise ValueError('Specify a valid minimization method, or leave as default')
    if method in nosupport_jacobian and use_jacobian:
        use_jacobian = False
        print('Jacobian not supported by this method, will ignore flag')
    working_obs = [itm[1:] for ob in list_of_tracklets for itm in ob]

    # Left over from calculating the mean of the times of the obs to get as new ref
    # t_ref_mean = sum(obs[0] for obs in working_obs)/len(working_obs)

    args = [(obs[5],obs[6],obs[7],obs[0]-obs[1]-t_ref,obs[2],obs[3]) for obs in working_obs]

    # Get the avg of the trackelt params
    x0_guess = []
    for trkl in list_of_tracklets:
        obs_in_trkl = [i[1:] for i in trkl]
        x0_guess.append(np.array(fit_tracklet(t_ref, g_init, gdot_init, obs_in_trkl)[:4]))
    x0_guess = np.append(np.array(x0_guess).mean(axis=0), [g_init,gdot_init])
  
    def loss(arr):
        """ Loss function: aggregate the errors from the loss of the theta_x and theta_y
        with equal weighting and minimize this function. Regretablly, this function
        uses the global variable (a list of lists) 'args' to aggregate the errors
        from each observation in realtion to both theta_x and theta_y.
        ---------
        Args: NOTE: uses the global variable 'args'
              arr; the array of the a, adot, b, bdot, g, gdot values we are
                    tuning to minimize this loss function.
        ---------
        Returns: float, the loss of the function in terms of the sum of
                    squared error for both theta_x and theta_y.
        """
        loss= 0.0
        a, adot, b, bdot, g, gdot = arr

        for arg in args:
            xe, ye, ze, t_emit, theta_x, theta_y = arg
            tx = (a + adot*t_emit - g*xe)/(1 + gdot*t_emit - g*ze)
            ty = (b + bdot*t_emit - g*ye)/(1 + gdot*t_emit - g*ze)
            loss+= (theta_x-tx)**2 + (theta_y-ty)**2

        return loss

    def loss_jacobian(arr):
        """ The analytical jacobian of the loss function.
        ---------
        Args: NOTE: uses the global variable 'args'
              arr; the array of the a, adot, b, bdot, g, gdot values we are
                    tuning to minimize this loss function.
        ---------
        Returns: numpy array length 6, the derivative of the loss function  with
                    respect to each of the 6 parameters.
        """
        md = np.array(args)
        L = md[:,4] #theta_x
        M = md[:,5] #theta_y
        t = md[:,3] #t_emit
        x = md[:,0]
        y = md[:,1]
        z = md[:,2]

        a,b,c,h,f,k = arr
        der_a = (2*(a+b*t-f*x+L*(-1-k*t+f*z)))/(1+k*t-f*z)**2
        der_b = (-2*t*(-a+L-b*t+k*L*t+f*x-f*L*z))/(1+k*t-f*z)**2
        der_c = (2*(c+h*t-f*y+M*(-1-k*t+f*z)))/(1+k*t-f*z)**2
        der_h = (-2*t*(-c+M-h*t+k*M*t+f*y-f*M*z))/(1+k*t-f*z)**2
        der_f = 2*(-(((a+b*t-f*x)*z)/(1+k*t-f*z)**2)+x/(1+k*t-f*z))*(L-(a+b*t-f*x)/(1+k*t-f*z))+2*(-(((c+h*t-f*y)*z)/(1+k*t-f*z)**2)+y/(1+k*t-f*z))*(M-(c+h*t-f*y)/(1+k*t-f*z))
        der_g = (2*t*(a+b*t-f*x)*(L-(a+b*t-f*x)/(1+k*t-f*z)))/(1+k*t-f*z)**2+(2*t*(c+h*t-f*y)*(M-(c+h*t-f*y)/(1+k*t-f*z)))/(1+k*t-f*z)**2

        return np.array([der_a.sum(), der_b.sum(), der_c.sum(), der_h.sum(), der_f.sum(), der_g.sum()])


    def loss_hessian(arr):
        md = np.array(args)
        L_arr = md[:,4] #theta_x
        M_arr = md[:,5] #theta_y
        t_arr = md[:,3] #t_emit
        x_arr = md[:,0]
        y_arr = md[:,1]
        z_arr = md[:,2]
        a,b,p,h,f,k = arr
        
        
        H_overall = np.zeros([6,6])
        #print(y.shape)
        for i in range(len(L_arr)):
            H = np.zeros([6,6])
        
            L = L_arr[i]
            M = M_arr[i]
            t = t_arr[i]
            x = x_arr[i]
            y = y_arr[i]
            z = z_arr[i]
            H[0,0]=2/(1+k*t-f*z)**2 #a,a
            H[0,1]=(2*t)/(1+k*t-f*z)**2 #a,b
            H[0,2]=0 #a,p
            H[0,3]=0 #a,h
            H[0,4]=(-2*(x*(1+k*t+f*z)+z*(-2*a+L-2*b*t+k*L*t-f*L*z)))/(1+k*t-f*z)**3 #a,f
            H[0,5]=(2*t*(-2*a+L-2*b*t+k*L*t+2*f*x-f*L*z))/(1+k*t-f*z)**3 #a,k
            H[1,1]=(2*t**2)/(1+k*t-f*z)**2 #b,b
            H[1,2]=0 #b,p
            H[1,3]=0 #b,h
            H[1,4]=(-2*t*(-(((a+b*t-f*x)*z)/(1+k*t-f*z)**2)+x/(1+k*t-f*z)))/(1+k*t-f*z)-(2*t*z*(L-(a+b*t-f*x)/(1+k*t-f*z)))/(1+k*t-f*z)**2 #b,f
            H[1,5]=(2*t**2*(-2*a+L-2*b*t+k*L*t+2*f*x-f*L*z))/(1+k*t-f*z)**3 #b,k
            H[2,2]=2/(1+k*t-f*z)**2 #p,p
            H[2,3]=(2*t)/(1+k*t-f*z)**2 #p,h
            H[2,4]=(2*(-y+M*z))/(1+k*t-f*z)**2+(4*z*(p+h*t-f*y+M*(-1-k*t+f*z)))/(1+k*t-f*z)**3 #p,f
            H[2,5]=(2*t*(M-2*p-2*h*t+k*M*t+2*f*y-f*M*z))/(1+k*t-f*z)**3 #p,k
            H[3,3]=(2*t**2)/(1+k*t-f*z)**2 #h,h
            H[3,4]=(-2*t*(-(((p+h*t-f*y)*z)/(1+k*t-f*z)**2)+y/(1+k*t-f*z)))/(1+k*t-f*z)-(2*t*z*(M-(p+h*t-f*y)/(1+k*t-f*z)))/(1+k*t-f*z)**2 #h,f
            H[3,5]=(2*t**2*(M-2*p-2*h*t+k*M*t+2*f*y-f*M*z))/(1+k*t-f*z)**3 #h,k
            H[4,4]=2*(-(((a+b*t-f*x)*z)/(1+k*t-f*z)**2)+x/(1+k*t-f*z))**2+2*((-2*(a+b*t-f*x)*z**2)/(1+k*t-f*z)**3+(2*x*z)/(1+k*t-f*z)**2)*(L-(a+b*t-f*x)/(1+k*t-f*z))+2*(-(((p+h*t-f*y)*z)/(1+k*t-f*z)**2)+y/(1+k*t-f*z))**2+2*((-2*(p+h*t-f*y)*z**2)/(1+k*t-f*z)**3+(2*y*z)/(1+k*t-f*z)**2)*(M-(p+h*t-f*y)/(1+k*t-f*z)) #f,f
            H[4,5]=(2*t*(a+b*t-f*x)*(-(((a+b*t-f*x)*z)/(1+k*t-f*z)**2)+x/(1+k*t-f*z)))/(1+k*t-f*z)**2+(4*t*(a+b*t-f*x)*z*(L-(a+b*t-f*x)/(1+k*t-f*z)))/(1+k*t-f*z)**3-(2*t*x*(L-(a+b*t-f*x)/(1+k*t-f*z)))/(1+k*t-f*z)**2+(2*t*(p+h*t-f*y)*(-(((p+h*t-f*y)*z)/(1+k*t-f*z)**2)+y/(1+k*t-f*z)))/(1+k*t-f*z)**2+(4*t*(p+h*t-f*y)*z*(M-(p+h*t-f*y)/(1+k*t-f*z)))/(1+k*t-f*z)**3-(2*t*y*(M-(p+h*t-f*y)/(1+k*t-f*z)))/(1+k*t-f*z)**2 #f,k
            H[5,5]=(2*t**2*(a+b*t-f*x)**2)/(1+k*t-f*z)**4+(2*t**2*(p+h*t-f*y)**2)/(1+k*t-f*z)**4-(4*t**2*(a+b*t-f*x)*(L-(a+b*t-f*x)/(1+k*t-f*z)))/(1+k*t-f*z)**3-(4*t**2*(p+h*t-f*y)*(M-(p+h*t-f*y)/(1+k*t-f*z)))/(1+k*t-f*z)**3 #k,k

            #enforce symmetry
            for i in range(0,6):
                for j in range(i+1,6):
                    H[j,i]=H[i,j]
                        
            H_overall+=H
        return H_overall

    min_options = {}
    if tol==0: #additional options for specific solvers to ensure max accuracy
        if method=='CG':
            min_options['gtol'] = 0
        if method in ['Powell','Newton-CG','TNC']:
            min_options['xtol'] = -1 if method=='Newton-CG' else 0
        if method in ['Powell','L-BFGS-B','TNC','SLSQP']:
            min_options['ftol'] = 0
        if method in ['CG','BFGS','L-BFGS-B','TNC','dogleg','trust-ncg']:
            min_options['gtol'] = 0
        if method in ['COBYLA']:
            min_options['tol'] = 0
            min_options['catol'] = 0

    use_hessian = False
    #automatically use Hessian if we use one of these methods...
    if method=='dogleg' or method=='trust-ncg' or method=='Newton-CG': use_hessian=True
    jc = loss_jacobian if use_jacobian else None
    hs = loss_hessian if use_hessian else None

    opt_out = minimize(loss,x0=np.array(x0_guess), method=method, tol=tol, options=min_options, jac=jc, hess=hs)
    
    nit = (opt_out.nit if 'nit' in opt_out else math.nan) # number of iterations in solver
    
    # calc error
    err,err_arr = error(opt_out.x,args,flag=flag)

    # number of evaluations of the objective function, error, jacobian, hessian
    def_return = [opt_out.x, opt_out.fun, err, err_arr]
    if details:
        def_return.append(nit)

    return tuple(def_return)

def cluster_months(fit_dicts, rad, GM=MPC_library.Constants.GMsun):
    """ This function will take in the fit_dict outputs of the postprocessing step
    and find then cluster the orbital elements (or parameters) based on a reference
    time.  This will hopefully product valid, month to month clusters.
    ---------     all the fit_dicts for every month
    Args: fit_dicts list of fit_dict objects where the key is string cluster_id and
                the value is the a tuple with EITHER the related, fitted a,adot,b,
                bdot,g,gdot parameters for that cluster OR the realted, transformed
                a, e, i, big_omega, little_omega, m from the orbital elements transform,
                and array of observation level errors.
          rad; float, the new radius within which to search for meta (month-to-month)
                    clusters
    ---------
    Returns: final_dict where the key is string cluster_id and the value is the a tuple with
                EITHER the related, fitted a,adot,b,bdot,g,gdot parameters for that
                cluster OR the realted, transformed a, e, i, big_omega, little_omega, m
                from the orbital elements transform, and array of observation level errors.
                The difference here is that the fit dict has been clustered and re-fit.
             final_dict_cid; a dict meant for plotting, which is keyed on tracklet id,
                and contains a numeric cluster key as a value.
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
            final_dict[cluster_key] = (np.array(orb_params).mean(axis=0), np.mean(orb_errors))

    return final_dict, get_cid_dict(final_dict, shared=False)


def postprocessing(infilename, clust_counter, pixels, nside, n, orb_elms=True, angDeg=5.5, gi=0.4, gdoti=0.0):
    """ This function will take in a cluster counter and find the right tracklets
    (elements of the largest cluster) and fit those clusters with our orbit fitting
    algorithm.  This step is postprocessing for each healpix window, and preps the
    fitted results to be transformed and compared across time frames.  The result of
    this function is all the cluster keys in the .trans file and the related fitted
    orbit parameters a,adot,b,bdot,g,gdot. When clusters share a tracklet (or more),
    we choose the cluster with the greater number of tracklets as the best cluster going
    forward.  After this step, transform the resulting orbital parameters to the
    "orbital elements" parameters, and compare to other months, and/or reference orbits.
    ----------
    Args: infilename; the location and filename of the .trans file the user wants to use
          clust_counter; the counter result of the previous run of find_clusters() or
                cluster_clusters().
          pixels; list or range, the range of all healpix sections the user wants.
          nside; int, number of sides in the healpix division of the sky
          n; int, the lunar center. use the function in utils to get the jd
          angDeg; float, the angle in degrees
          gi; float; the initial, asserted gamma value (distance from observer to the asteroid)
          gdoti; float; the initial, asserted gamma dot value of radial velocity.
    ----------
    Returns: fit_dict where the key is cluster_id and the value is the a tuple with
                EITHER the related, fitted a,adot,b,bdot,g,gdot parameters for that
                cluster OR the realted, transformed a, e, i, big_omega, little_omega, m
                from the orbital elements transform, and array of observation level errors.
     """
    cid_dict = {} #get_cid_dict(clust_counter, shared=False)
    helper = {}
    for str_cid in clust_counter.keys():
        for tid in str_cid.split('|'):
            if tid in cid_dict.keys() and helper[tid]<len(str_cid.split('|')):
                cid_dict[tid] = str_cid
                helper[tid] = len(str_cid.split('|'))
            else:
                cid_dict[tid] = str_cid
                helper[tid] = len(str_cid.split('|'))
    # cid_dict now has tracklet id as key and joined cluster id as value

    res_dicts = get_res_dict(infilename, pixels, nside, n, angDeg=angDeg, g=gi, gdot=gdoti)
    t_ref = util.lunation_center(n)
    fit_dict= {}

    # For each chunk of sky in our window
    for pix, results_d in res_dicts.items():

        # referencd vector for transform
        r_ref = hp.pix2vec(nside, pix, nest=True)

        agg_dict = defaultdict(list)
        for tid, v in cid_dict.items():
            if results_d[tid]!=[]:
                agg_dict[v].append([tuple([tid]+list(obs)) for obs in results_d[tid]])

        # k is the str cluster id, v is the tracklets in the cluster id
        for k, v in agg_dict.items():
            params, func_val, chisq, chiarr = full_fit_t_loss(t_ref, gi, gdoti, v)

            # perform transform from pbasis to orbital elements
            if orb_elms:
                params = util.pbasis_to_elements(params, r_ref)

            if k not in fit_dict.keys():
                fit_dict[k] = (params, chiarr)
            else:
                if len(fit_dict[k][1])<len(chiarr):
                    fit_dict[k] = (params, chiarr)

    return fit_dict


def cluster_clusters(infilename, clust_count, pixels, nside, n, dt=15., rad=0.00124, new_rad=0.00124, angDeg=5.5, gi=0.4, gdoti=0.0):
    """ This function takes in a previous run of find_clusters() in the form of a cluster_counter
    object.  It will fit orbits to each cluster in the group and compare the resulting
    a,adot,b,bdot,g,gdot (yes, all 6 parameters) with a KDTree and choose the clusters that
    are within a specified radius of each other as matches and join those two clusters.

    Conceptually, this is meant to get rid of the problem at the lower level where we
    had lots of sub clusters in a large cluster of tracklets because the subclusters
    were a little tighter than the group as a whole.  This method is designed to cluster
    (as one cluster) those instances.
    ---------
    Args: infilename; the location and filename of the .trans file the user wants to use
          clust_count; the counter result of the previous run of find_clusters()
          pixels; list or range, the range of all healpix sections the user wants.
          nside; int, number of sides in the healpix division of the sky
          n; int, the lunar center. use the function in utils to get the jd
          dt; the best dt value, currently dt=15 based on first principles
          rad; the best rad value, based on dt=15 we calculated rad to be 0.00124
                so that is the default value.
          new_rad; the radius we use to cluster once we transform the approximately
                    close points with the fitted g and gdot.  Defaults to 0.00124.
          angDeg; float, the angle in degrees
          gi; float; the initial, asserted gamma value (distance from observer to the asteroid)
          gdoti; float; the initial, asserted gamma dot value of radial velocity.
    -------------
    Returns: cluster_counter; Counter() object with concatenated tracklet_id's with '|'
                as the key and the associated count,
             cluster_id_dict, dictionary object with the tracklet id's as keys and the
                cluster_id's as values.  This result comes from the get_cid_dict function.
                see that function's documentation for better description."""
    res_dicts = get_res_dict(infilename, pixels, nside, n, angDeg=angDeg, g=gi, gdot=gdoti)
    t_ref = util.lunation_center(n)
    cluster_counter = Counter()

    # For each chunk of sky in our window
    for pix, results_d in res_dicts.items():

        points, labels = [], []

        fit_dict = _nlin_fits(clust_count, results_d, gi, gdoti, t_ref)
        # print(len(clust_count.keys()), len(fit_dict.keys()))
        for k,v in fit_dict.items():
            points.append(v[0])
            labels.append(k)

        points = np.array(points)
        if len(points)<2:
            continue
        tree = scipy.spatial.cKDTree(points)
        matches = tree.query_ball_tree(tree, new_rad)

        for j, match in enumerate(matches):
            cluster_list =[]
            for idx in match:
                c_id = labels[idx]
                cluster_list.extend(c_id.split('|'))
            cluster_key='|'.join(sorted(cluster_list))
            cluster_counter.update({cluster_key: 1})

    return cluster_counter, get_cid_dict(cluster_counter,shared=False)

def get_cid_dict(cluster_counter,shared=False,string=False):
    """ This function takes in a cluster_counter object from a result of
    find_clusters() or cluster_clusters().  The purpose of this function is to
    make a strategic choice of how to display or organize overlapping clusters.
    The shared tag, will assign a numeric cluster id (for graphing) to each
    tracklet that appears only once in a cluster, and a common "cluster id" of
    -42 to each cluster that is shared by two or more clusters. If the shared
    option is switched off, clusters with more elements will always superseed
    smaller clusters when both clusters claim a shared tracklet. Repeated below.

    NOTE:  If the shared option is switched off, clusters with more elements will
    always superseed smaller clusters when both clusters claim a shared tracklet!
    ---------
    Args: cluster_counter; the cluster_counter object from a result of
            find_clusters() or cluster_clusters().
          shared; bool, a flag to determine whether the id's are calculated with
                    a fixed shared value for all shared tracklets, or ordered
                    where each tracklet automatically belongs to the biggest
                    cluster it is a part of when it is part of more that 1 cluster.
    ---------
    Returns: cluster_id_dict; a dictionary where the key is tracklet id and
                the values are a numeric cluster id (used for visualization).
    """
    cluster_id_dict = {}
    helper = {}
    for i, str_cid in enumerate(cluster_counter.keys()):
        for tid in str_cid.split('|'):
            if shared:
                if tid in cluster_id_dict.keys():
                    cluster_id_dict[tid] = -42
                else:
                    cluster_id_dict[tid] = i
            else:
                if tid in cluster_id_dict.keys() and helper[tid]<len(str_cid.split('|')):
                    if string:
                        cluster_id_dict[tid] = str_cid
                    else:
                        cluster_id_dict[tid] = i
                    helper[tid] = len(str_cid.split('|'))
                else:
                    if string:
                        cluster_id_dict[tid] = str_cid
                    else:
                        cluster_id_dict[tid] = i
                    helper[tid] = len(str_cid.split('|'))

    return cluster_id_dict

def get_cluster_level_dicts(clust_ids,trans_path,nside,n,pixel,g=0.4,gdot=0.0):
    """ This function takes in a dict keyed on tracklet id and with cluster id
    as values.  It returns the 'agg_dict' data structure which has cluster id as
    a key and a list of the associated tracklets with a tuple containing
    tracklet_id,a,adot,b,bdot.
    --------
    Args: clust_ids; dict keyed on tracklet id and values as cluster id.
            output from the find_clusters function.
          trans_path, str, the path related the .trans file in question (same
            path as for the find_clusters function).
          nside; int, number of sides for the healpix division
          n, int, the index for the lunation center
          pixel, int, the specific pixel the user wants to look at, ranges from
            -825 to 14.
    --------
    Returns: agg_dict, par_results, chi_res, where agg_dict is a dict has cluster id as
        a key and a list of the associated tracklets with a tuple containing
        tracklet_id,a,adot,b,bdot; par_results is a dictionary keyed on cluster id and
        with the resulting a,adot,b,bdot,g,gdot fitted orbital parameters; chi_res
        is a dict keyed on cluster id with the related error terms for each cluster
        as values.
    """
    res_dict = get_res_dict(trans_path,[pixel],nside,n)[pixel]

    agg_dict = defaultdict(list)
    for k,v in clust_ids.items():
        # k is the tracklet id, v is the cluster id
        agg_dict[v].append([tuple([k]+list(i)) for i in res_dict[k]])

    par_results,chi_res  = {},{}
    po = [k for k,v in agg_dict.items() if v!=[]]
    for idx in po:
        all_res = full_fit_t_loss(util.lunation_center(n), g, gdot, agg_dict[idx])
        par_results[idx] = all_res[0]
        chi_res[idx] = all_res[2]

    return agg_dict, par_results, chi_res

def fit_extend(infilename, clust_count, pixels, nside, n, dt=15., rad=0.00124, new_rad=0.00124, angDeg=5.5, gi=0.4, gdoti=0.0):
    """  This function takes a whole window of time (with dt=15 about a month)
    and calculates the fitted orbits for each cluster already clustered in the
    passed file.  Then we compare the old transform values (the ones we clustered on
    in the first run) and pick a arbitrarily large radius (the only limitation on radius
    size here is computational efficiency because we are making a new KDTree with the
    results).

        So, for each cluster check the nearness of the surrounding points in
    the old transform and pick canidates (with an arb large radius) to get transformed
    with the new g and gdot (that come from our cluster orbit fits), and then compare
    again, with new radius (defaults to the initial best radius). Then if new
    cluster id's have been added to the cluster via this method, add them to
    cluster_counter and change the realted tracklet id in cluster_id_dict (newer
    clusters superseed older in the case of overlap (per our standard protocol).
    ------------
    Args: infilename; the location and filename of the .trans file the user wants to use
          clust_count; the counter result of the previous run of find_clusters()
          pixels; list or range, the range of all healpix sections the user wants.
          nside; int, number of sides in the healpix division of the sky
          n; int, the lunar center. use the function in utils to get the jd
          dt; the best dt value, currently dt=15 based on first principles
          rad; the best rad value, based on dt=15 we calculated rad to be 0.00124
                so that is the default value.
          new_rad; the radius we use to cluster once we transform the approximately
                    close points with the fitted g and gdot.  Defaults to 0.00124.
          angDeg; float, the angle in degrees
          gi; float; the initial, asserted gamma value (distance from observer to the asteroid)
          gdoti; float; the initial, asserted gamma dot value of radial velocity.
    -------------
    Returns: cluster_counter; Counter() object with concatenated tracklet_id's with '|'
                as the key and the associated count,
             cluster_id_dict, dictionary object with the tracklet id's as keys and the
                cluster_id's as values.  This result comes from the get_cid_dict function.
                see that function's documentation for better description.
    """
    res_dicts = get_res_dict(infilename, pixels, nside, n, angDeg=angDeg, g=gi, gdot=gdoti)
    t_ref = util.lunation_center(n)
    cluster_counter = Counter()

    # For each chunk of sky in our window
    for pix, results_d in res_dicts.items():

        # Get the arrows with the old transforms (this is also kind of redundent, but pmo is roe so leave for now)
        ot_arrows = list(_return_arrows_resuts(results_d, t_ref, [(gi,gdoti)], \
                                            fit_tracklet_func=fit_tracklet).values())[0]
        i = 0
        ot_label_dict={}
        combined=[]
        for aro in ot_arrows:
            k, cx, mx, cy, my, t = aro
            ot_label_dict[i] = k
            combined.append([cx, mx*dt, cy, my*dt])
            i +=1

        ot_points=np.array(combined)
        ot_tree = scipy.spatial.cKDTree(ot_points)

        # Get the nonlinear fit of the clusters in this pixel
        fit_dict = _nlin_fits(clust_count,results_d,gi,gdoti,t_ref)

        # Note: Read the docs for explaination of this procedure
        for k,v in fit_dict.items():
            # k is the string cluster id and v is the fitted 6 params, fval, err, and arr_err
            params = v[0]
            a,adot,b,bdot,g,gdot = params
            trkl_ids_in_cluster = set(k.split('|'))
            canidates = ot_tree.query_ball_point(params[:4],r=rad*50.) # tuneable param

            if canidates !=[]:
                nt_points = []
                nt_label_dict = []
                for idx in canidates:
                    tracklet_id = ot_label_dict[idx].strip()
                    nt_a,nt_ad,nt_b,nt_bd = fit_tracklet(t_ref, g, gdot, results_d[tracklet_id])[:4]
                    nt_points.append(np.array((nt_a, nt_ad*1., nt_b, nt_bd*1.))) # NOTE: Not scaling by dt here
                    nt_label_dict.append(tracklet_id)

                nt_tree = scipy.spatial.cKDTree(nt_points)
                matches = nt_tree.query_ball_point(params[:4],r=new_rad)

                cluster_list =[]
                for idx in matches:
                    tracklet_id = nt_label_dict[idx].strip()
                    cluster_list.append(tracklet_id)

                trkl_ids_in_cluster |= set(cluster_list)
                cluster_key='|'.join(sorted(list(trkl_ids_in_cluster)))
                cluster_counter.update({cluster_key: 1})

    return cluster_counter, get_cid_dict(cluster_counter,shared=False)

def _nlin_fits(clust_count, results_d, g_init, gdot_init, t_ref):
    """ This is a helper function for the fit_extend function.
    It calculates the fitted parameters for each cluster it is given in
    agg_dict, and returns a dictionary with cluster_id as the key and the realted
    parameters as the value.
    NOTE: agg_dict is a dictionary with cluster id (string) as key and list of lists
            of tuples where the outer list represents the cluster, the inner list
            represents a tracklet and each tuple represents an observation as values.
    -------
    Args: clust_ids; a dictionary with tracklet id as key and cluster id as values
          results_d; a dict where the key is the tracklet id and the value is
                      (jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze).
          g_init; float, the gamma value (distance from observer to the asteroid)
          gdot_init; float, the gamma dot value of radial velocity.
          t_ref; the lunar center of the month in question.
    -------
    Returns: fit_dict where the key is cluster_id and the value is the a tuple with
                the related, fitted a,adot,b,bdot,g,gdot parameters for that cluster,
                obj function value, error value, and array of observation level errors.
    """
    # Create agg_dict for the specific chunk of sky
    agg_dict = defaultdict(list)
    for i, str_cid in enumerate(clust_count.keys()):
        # i will be our assigned cluster id and k is the | joined tracklet id
        for tid in str_cid.split('|'):
            # The possibility of this exists only because results_d is a default dict (otherwise keyerror)
            if results_d[tid]!=[]:
                agg_dict[str_cid].append([tuple([tid]+list(obs)) for obs in results_d[tid]])

    fit_dict= {}
    # k is the str cluster id, v is the tracklets in the cluster id
    for k, v in agg_dict.items():
        params, func_val, chisq, chiarr = full_fit_t_loss(t_ref, g_init, gdot_init, v)
        fit_dict[k] = (params, func_val, chisq, chiarr)

    return fit_dict

def get_res_dict(infilename, pixels, nside, n, angDeg=5.5, g=0.4, gdot=0.0):
    """ Function to get the results dict object from a given file name. The
    Results dict object is a dict with tracklet id as the key and the jd and
    delta jd (Julian data) theta x,y,z, and xe,ye,ze as values in a tuple.
    --------
    Args: infilename; str, the name of the .trans file in question.
          pixels; list or range, the range of all healpix sections the user wants.
          nside; int, number of sides in the healpix division of the sky
          n; int, the lunar center. use the function in utils to get the jd
          angDeg; float, the angle is degrees
          g; float; the gamma value (distance from observer to the asteroid)
          gdot; float; the gamma dot value of radial velocity.
    -------
    Returns: master_dict; a dictionary with a results dictionary at each key
                (key in unique on pixel).
    """
    hp_dict = util.get_hpdict(infilename)

    master_dict = {}
    for i in pixels:
        vec = hp.pix2vec(nside, i, nest=True)
        neighbors = hp.query_disc(nside, vec, angDeg*np.pi/180., inclusive=True, nest=True)
        lines = []

        for pix in neighbors:
            for line in hp_dict[pix]:
                lines.append(line)
        if len(lines) > 0:
            res_dict = get_tracklet_obs(vec,lines)
            master_dict[i] = res_dict

    return master_dict

def get_tracklet_obs(vec,lines):
    """ This function gets the results dict given the vector and the lines of
    the input file. Also of note, mat is a rotation matrix that converts from
    ecliptic vectors to the projection coordinate system. The projection
    coordinate system has z outward, x parallel to increasing ecliptic longitude,
     and y northward, making a right-handed system.
    ---------
    Args: vec; list, is the reference direction in ecliptic coordinates
          lines; all the lines from a *.trans file.
    ---------
    Returns: results dict where the key is the tracklet id and the value is
                (jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze).
    """
    results_dict = defaultdict(list)

    vec = np.array(vec)
    vec = vec/np.linalg.norm(vec)

    mat = util.xyz_to_proj_matrix(vec)

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

            # Rotate to projection coordinates
            theta_x, theta_y, theta_z = np.dot(mat, r_target)

            # Ignore theta_z after this; it should be very nearly 1.
            # Get observatory position, ultimately in projection coordinates.
            x_obs, y_obs, z_obs = line[98:138].split()
            r_obs = np.array([float(x_obs), float(y_obs), float(z_obs)])

            # Rotate to projection coordinates
            xe, ye, ze = np.dot(mat, r_obs)

            # This is the light travel time
            dlt = ze/MPC_library.Constants.speed_of_light

            # Append the resulting data to a dictionary keyed on trackletID.
            results_dict[trackletID.strip()].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))

    return results_dict

def _return_arrows_resuts(results_dict, t_ref, g_gdot_pairs, fit_tracklet_func):
    """Now that we have the observations for each tracklet gathered together,
    we iterate through the tracklets, doing a fit for each one. This fit is just
    over the a, adot, b, bdot parameters.
    --------
    Args: Results_dict; see the get_tracklet_obs() function.
          t_ref; the lunar center of the month in question.
          g_gdot_pairs; list of tuples of g,gdot pairs.
          fit_tracklet_func; the tracklet fitting specified above at the higher
          level.
    --------
    Returns: master_results, where master results is a array of all points related
                to the ggdot pair that is the key of this dict.
    """

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

def _write_arrows_files(results_dict, t_ref, g_gdot_pairs, fit_tracklet_func, vec, outfilename, cluster_id_dict={}):
    """Now that we have the observations for each tracklet gathered together,
    we iterate through the tracklets, doing a fit for each one. This fit is just
    over the a, adot, b, bdot parameters.
    --------
    Args: Results_dict; see the get_tracklet_obs() function.
          t_ref; the lunar center of the month in question.
          g_gdot_pairs; list of tuples of g,gdot pairs.
          fit_tracklet_func; the tracklet fitting specified above at the higher
          level.
          vec; list, is the reference direction in ecliptic coordinates
          outfilename; str, pretty self explanatory, this is the name of the file output
          cluster_id_dict; dict, the dictionary accompanying the results of find_clusters
            that has keys as trackletID and values as cluster id.
    --------
    Returns: None; writes results out to files.
    """
    for g_gdot in g_gdot_pairs:
        g, gdot = g_gdot
        results = []
        for k, v in results_dict.items():

            cx, mx, cy, my, t0 = fit_tracklet_func(t_ref, g, gdot, v)
            if cluster_id_dict=={}:
                outstring = "%12s %16.9lf %16.9lf %16.9lf %16.9lf %16.9lf\n" % (k, cx, mx, cy, my, t0)
            else:
                try:
                    cid = cluster_id_dict[k.strip()]
                except KeyError:
                    cid = -1
                outstring = "%12s %16.9lf %16.9lf %16.9lf %16.9lf %d\n" % (k, cx, mx, cy, my, cid)

            results.append(outstring)

        if len(results)>0:
            with open(outfilename, 'w') as outfile:
                outstring = '#  g = %lf\n' % (g)
                outfile.write(outstring)
                outstring = '#  gdot = %lf\n' % (gdot)
                outfile.write(outstring)
                outstring = '#  vec= %lf, %lf, %lf\n' % (vec[0], vec[1], vec[2])
                outfile.write(outstring)
                if cluster_id_dict=={}:
                    outstring = '#  desig              alpha         alpha_dot       beta             beta_dot         dt \n'
                else:
                    outstring = '#  desig              alpha         alpha_dot       beta             beta_dot         clust_id \n'
                outfile.write(outstring)
                for outstring in results:
                    outfile.write(outstring)

def transform_to_arrows(t_ref, g_gdot_pairs, vec, lines, outfilename='',makefiles=False, \
                            cluster_id_dict={},fit_tracklet_func=fit_tracklet):
    """ We are going to re-transform those assuming a fixed z (or gamma) value
    with respect to the sun and the reference direction, rather than a
    fixed r, at the reference time. Rotate observatory positions to projection coordinates,
    and recalculate simple z-based light-time correction. Rotate the observations
    to projection coordinates, but they will be theta_x, theta_y only

    Fit the simple abg model, for fixed gamma and
    possibly gamma_dot. It takes a reference time (t_ref), a set of z, zdot pairs
    (z_zdot_pairs), a reference direction vector (vec), and a set of observation
    lines that have been selected for a region of sky and time slice (lines)
    -------
    Args: t_ref, float, the lunation center,
          g_gdot pairs is a list of tuples with the g, gdot pairs to use
                for the select_clusters_z functionality, pass the g, gdot in
                [(g,gdot)] format
        fit_tracklet_func; the tracklet fitting specified above at the higher
        level.
        vec; list, is the reference direction in ecliptic coordinates
        outfilename; str, pretty self explanatory, this is the name of the file output (none in this case)
        NA: cluster_id_dict; dict, the dictionary accompanying the results of find_clusters
          that has keys as trackletID and values as cluster id.
    --------
    Returns: master_results, where master results is a array of all points related
                to the ggdot pair that is the key of this dict.
    """
    results_dict = get_tracklet_obs(vec,lines)
    if makefiles:
        _write_arrows_files(results_dict, t_ref, g_gdot_pairs, fit_tracklet_func, vec, outfilename,cluster_id_dict)
    else:
        return _return_arrows_resuts(results_dict, t_ref, g_gdot_pairs, fit_tracklet_func)


def cluster_sky_regions(g_gdot_pairs, pixels, t_ref, infilename, nside=8, angDeg=7.5, cluster_func=transform_to_arrows):
    """ This function is the main cluster function.  We take in lots of arguments
    and return a dictionary with keys as pixels (sections in the healpix breakup of the
    sky), and the values as the output (master_dict) from transform_to_arrows. This
    is made up of a dict of keys as g_gdot pairs and results arrays as values with
    all the related tracklets.
    --------
    Args: t_ref, float, the lunation center,
          g_gdot pairs is a list of tuples with the g, gdot pairs to use
                for the select_clusters_z functionality, pass the g, gdot in
                [(g,gdot)] format
          pixels; range or list, the healpix breakup of the sky
          infilename; str, the location and filename of the .trans file
          nside; int, the number of sides in the healpix pixels breakout
          angDeg; float, the angle is degrees
          cluster_func; function to use for clustering.
    ---------
    Returns: a dictionary with pixels as keys and dicts as values where the inner
                dicts have g_gdotpairs as keys and realted points as values.
    """
    hp_dict = util.get_hpdict(infilename)

    pixel_results = {}
    for prog,i in enumerate(pixels):
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



def _get_cluster_counter(master, dt, rad, mincount):
    """ This function gets the cluster_counter object. The object is unique on
    clusters and the cluster_id_dict is unique on tracklets.  The cluster_counter
    has tracklet_id's joined with pipes '|' as keys and the number of occurances
    as values. The cluster_id_dict has tracklet_id as the key and the cluster id
    as the value of the dict.
    ---------
    Args: master; the results of the clustering function, which returns a dict
            with pixels as keys and dicts as values where the inner dicts have
            g_gdot_pairs as keys and realted points as values.
          dt; float, the dt measure we set.  First principles suggests 15, but it is
                functionally the weight of the importance of position and velocity,
                so it can be whatever you want.
          rad; float, the cluster radius we look within for the KDTree.
          mincount; int, the fewest number of allowable trackelts is a cluster.
    ---------
    Returns:  cluster_counter dict with tracklet_id's joined with pipes '|' as
        keys and the number of occurances as values, and cluster_id_dict has
        tracklet_id as the key and the cluster id as the value of the dict.
    """
    cluster_counter = Counter()
    for pix, d in master.items():
        for g_gdot, arrows in d.items():

            i = 0
            label_dict={}
            combined=[]
            for k, cx, mx, cy, my, t in arrows:
                label_dict[i] = k
                combined.append([cx, mx*dt, cy, my*dt])
                i +=1
            points=np.array(combined)

            tree = scipy.spatial.cKDTree(points)
            matches = tree.query_ball_tree(tree, rad)

            for j, match in enumerate(matches):
                if len(match)>=mincount:
                    cluster_list =[]
                    for idx in match:
                        tracklet_id = label_dict[idx].strip()
                        cluster_list.append(tracklet_id)
                    cluster_key='|'.join(sorted(cluster_list))
                    cluster_counter.update({cluster_key: 1})

    return cluster_counter, get_cid_dict(cluster_counter,shared=False)

def _rates_to_results(rates_dict, dts):
    """ This function is a helper for train clusters, and it is used in the
    train_clusters method. It creates a data structure that is keyed on the dt
    value used in training, and as values, had a tuple with 4 components, a list
    of radius values, a list of the number of clusters at that radius value, a
    list of the number of errors at that radius value, and a list of all the keys
    associated with the cluster.
    --------
    Args: rates_dict; dict, a data stricture keyed on (dt,rad) tuple with the
            list of string cluster id's and number of errors stored as a tuple.
          dts; list, a list of the different dt values used in training =.
    --------
    Returns:  dict; a dictionary that is keyed on the dt value used in training,
            and as values, had a tuple with 4 components, a list of radius values,
            a list of the number of clusters at that radius value, a list of the
            number of errors at that radius value, and a list of all the keys
            associated with the cluster.
    """
    results_dict = {}
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

# Default for training clusters
############################################################################
gs =[0.4]
gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]
############################################################################

def train_clusters(pixels, infilename, t_ref, g_gdots=g_gdots,
                    dts=np.arange(5, 30, 5), radii=np.arange(0.0001, 0.0100, 0.0001),
                    cluster_sky_function=cluster_sky_regions, mincount=3):
    """ This function performs a training run over a set of training data provided
    by Matt Payne of the MPC in December 2017. This function allows the user to
    pick a range of dt and rad (specified by dts and radii) and calculate error rates
    over that grid and discover what the best dt and rad values are likely to be
    for the rest of the data.
    --------
    Args: pixels; range or list, the healpix breakup of the sky;
          infilename; str, the location and name of the .trans file the user
                wants to test.
          t_ref; the lunation center
          g_gdots; the gamma and gamma dot values the user specifies for the cluster
                    finding.
          dts; array, the dts measure the user wants to test.  First principles
                suggests 15, but it is functionally the weight of the importance
                of position and velocity, so it can be whatever you want.
          radii; array, the cluster radii values we look within for the KDTree search.
          cluster_sky_function; function used to cluster.
          mincount; int, the fewest number of allowable tracklets in a cluster.
    --------
    Returns: dictionary of the form, key is the dt value (one for each value in dts),
                and the value is a tuple of the form (ds, nclusters, nerrors, keys)
                where ds is the radius, nclusters is the number of successfully
                clustered clusters, nerrors is the number of incorrectly
                clustered clusters, and keys is the string of all tracklet_id's
                in a cluster joined with the '|' symbol.
    """
    master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)
    # The training case
    rates_dict={}
    for dt in dts:
        for rad in radii:
            cluster_counter, cluster_id_dict = _get_cluster_counter(master, dt, rad, mincount)
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


def test_clusters(pixels, infilename, t_ref, g_gdots=g_gdots,
                    dt=15, rad=0.00124,
                    cluster_sky_function=cluster_sky_regions, mincount=3):
    """ Also based on the training set.  This function allows the user to test
    out how well their chosen dt ad rad hyperparemeters perform on training data.
    --------
    Args: pixels; range or list, the healpix breakup of the sky;
          infilename; str, the location and name of the .trans file the user
                wants to test.
          t_ref; the lunation center
          g_gdots; the gamma and gamma dot values the user specifies for the cluster
                    finding.
          dt; float, the dt measure we set.  First principles suggests 15, but it is
                functionally the weight of the importance of position and velocity,
                so it can be whatever you want.
          rad; float, the cluster radius we look within for the KDTree.
          cluster_sky_function; function used to cluster.
          mincount; int, the fewest number of allowable tracklets in a cluster.
    --------
    Returns: tuple of form (int, int, list, list) where the first two ints are
                the number of correct clusters and incorrect clusters (errors)
                respectively, and the lists are the tracklet id's joined with '|'
                of the successes and the tracklet id's joined with '|'
                of the failures, respectively.
    """
    master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)

    cluster_counter, cluster_id_dict = _get_cluster_counter(master, dt, rad, mincount)
    test_set = list(cluster_counter.keys())
    success_dict, failure_counter = unique_clusters(test_set)
    return len(success_dict), len(failure_counter), list(success_dict.keys()), list(failure_counter.keys())

def find_clusters(pixels, infilename, t_ref, g_gdots=g_gdots,
                    dt=15, rad=0.00124,
                    cluster_sky_function=cluster_sky_regions, mincount=3):
    """ This is one of the centerpeice functions of the module.  It takes in the
    hyper parameters, in addition to the various g_gdots the user wants to try,
    and returns the cluster_counter and cluster_id_dict objects.  The cluster_counter
    has tracklet_id's joined with pipes '|' as keys and the number of occurances
    as values. The cluster_id_dict has tracklet_id as the key and the cluster id
    as the value of the dict.
    ---------
    Args: pixels; range or list, the healpix breakup of the sky;
          infilename; str, the location and name of the .trans file the user
                wants to test.
          t_ref; the lunation center
          g_gdots; the gamma and gamma dot values the user specifies for the cluster
                    finding.
          dt; float, the dt measure we set.  First principles suggests 15, but it is
                functionally the weight of the importance of position and velocity,
                so it can be whatever you want.
          rad; float, the cluster radius we look within for the KDTree.
          cluster_sky_function; function used to cluster.
          mincount; int, the fewest number of allowable tracklets in a cluster.
    ----------
    Returns: the cluster_counter and cluster_id_dict objects (mentioned above)
    """
    master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)

    cluster_counter, cluster_id_dict = _get_cluster_counter(master, dt, rad, mincount)
    return cluster_counter, cluster_id_dict


def output_sky_regions(pixels, infilename, nside=8, angDeg=7.5):
    """ Just gets the lines of a file for the pixels specified.
    -------
    Args: pixels; list or range, the range of all healpix sections the user wants.
          infilename; str, the name of the .trans file that you want to genreate files for.
          nside; int, number of sides in the healpix division of the sky
          angDeg; float; the angle in degrees.
    -------
    Return: list; the lines of the .trans file passed realted to the pixels specified.
    """
    hp_dict = util.get_hpdict(infilename)

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
    """ This function counts all the clusters, and returns several counters.
    The first return is clusterID_set which is just a set of the unique cluster ids.
    The second reutrn is mergedCounter_dict which is a Counter() keyed on pixel
    with values as another Counter() keyed on tracklet stem (the part that truly
    binds realted tracklets in the training set). The third return is mergedTime_dict
    which is keyed on pixel and contains as value another dictionary keyed on tracklet
    stem with a list of the time of occurance (in julian date) as values.
    -------
    Args: pixels; list or range, the range of all healpix sections the user wants.
          infilename; str, the name of the .trans file that you want to genreate files for.
          mincount; int, the minimum number of tracklets that must be present in a
                cluster for us to indentify it as a cluster.
    -------
    Return: a tuple with three values; first, clusterID_set which is just a set
                of the unique cluster ids. Second, mergedCounter_dict which is a
                Counter() keyed on pixel with values as another Counter() keyed
                on tracklet stem. Third, mergedTime_dict which is keyed on pixel
                and contains as value another dictionary keyed on tracklet stem
                with a list of the time of occurance as values.
    """
    true_counts={}
    mergedCounter_dict = {}
    mergedTime_dict = {}
    for pix in pixels:
        lines = output_sky_regions([pix], infilename=infilename)
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

    clusterID_set = set()
    for pix, cntr in mergedCounter_dict.items():
        for cID, count in cntr.items():
            if count>=mincount:
                clusterID_set.update({cID})

    return clusterID_set, mergedCounter_dict, mergedTime_dict

def member_counts(k, sep='|', suff='_'):
    """ This function counts the number of unique stems in a cluster by parsing
    for the stems, and making a Counter() object based on the parsed stem data.
    -------
    Args: k; the cluster id (made up of tracklet ids joined with '|')
          sep; str, the string to seperate by
          suff_ the seperating suffix between the initial part linking related
                tracklets and the suffix distinguishing tracklets of the same
                object.
    -------
    Returns: Counter() object keyed on the stems of a cluster.
    """
    keys = k.split(sep)
    stems = [key.split(suff)[0] for key in keys]
    stem_counter = Counter(stems)
    return stem_counter

def unique_clusters(test_set):
    """ This function defines the number of successes and failures in a cluster.
    The definition had changed somewhat from this definition to the evalu() definition,
    however, since this was originally used to tune the dt parameters, and is closely
    related (though less granular) than what we use now, it is retained.
    ---------
    Args: test_set; list of k cluster_ids
    --------
    Returns: success_dict, failure_counter where the first is a dictionary keyed
                on file stem (the common value for truly related tracklets), with
                values as a tuple of the (number of tracklets, cluster_id), and the
                second is a Counter() keyed on cluster_id of the failures.
    """
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

def generate_sky_region_files(infilename, pixels, nside, n, angDeg=5.5, g=0.4, gdot=0.0, cluster_id_dict={}):
    """ This function generates the files needed to plot the quiver functions.
    --------
    Args: infilename; str, the name of the .trans file that you want to genreate files for.
          pixels; list or range, the range of all healpix sections the user wants.
          nside; int, number of sides in the healpix division of the sky
          n; int, the lunar center. use the function in utils to get the jd
          angDeg; float, the angle is degrees
          g; float; the gamma value (distance from observer to the asteroid)
          gdot; float; the gamma dot value of radial velocity.
          cluster_id_dict; dict, the cluster_id_dict output from find clusters,
                     used strictly for plotting.
    --------
    Returns: None, but writes files."""
    hp_dict = util.get_hpdict(infilename)

    for i in pixels:
        vec = hp.pix2vec(nside, i, nest=True)
        neighbors = hp.query_disc(nside, vec, angDeg*np.pi/180., inclusive=True, nest=True)
        lines = []
        for pix in neighbors:
            for line in hp_dict[pix]:
                lines.append(line)
        if cluster_id_dict=={}:
            outfilename = infilename.rstrip('.trans') + '_hp_' + ('%03d' % (i)) + '_g'+ ('%.2lf' % (g))+'_gdot' + ('%+5.1le' % (gdot))
        else:
            outfilename = infilename.rstrip('.trans') + '_hp_' + ('%03d' % (i)) + '_g'+ ('%.2lf' % (g))+'_gdot' + ('%+5.1le' % (gdot))+'_cid'
        if len(lines) > 0:
            transform_to_arrows(util.lunation_center(n), [(g, gdot)], vec, lines, outfilename, makefiles=True, cluster_id_dict=cluster_id_dict)
