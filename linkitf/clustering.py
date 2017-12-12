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


# results_dict[trackletID].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))
def kbo2d_linear(pin, obs, t_ref):
    """Linearized version of the 2d projection of orbital position.  Only
       leading-order terms for each parameter's derivative are given.
       The derivative with respect to gdot is inserted here - note that
       it is much smaller than the others, and is the only part of the
       derivatives that has any dependence upon the incoming PBASIS.
    """
    jd_tdb, dlt, theta_x,theta_y,theta_z,xe,ye,ze = obs
    a,adot,b,bdot,g,gdot = pin
    t_emit = obs[0]-obs[1]-t_ref
    #Account for light-travel time differentially to leading order
    #by retarding KBO motion by z component of Earth's position.
    #Note: ignoring acceleration here.
    # t_emit = obs->obstime - ze/SPEED_OF_LIGHT;

    x = a + adot*t_emit - g * xe
    - gdot * (adot*t_emit*t_emit - g*xe*t_emit)
    y = b + bdot*t_emit - g * ye
    - gdot * (bdot*t_emit*t_emit - g*ye*t_emit)

    dx=np.zeros(6)
    dy=np.zeros(6)

    dx[1] = dy[3] = 1.
    dx[2] = dy[4] = t_emit
    dx[5] = -xe
    dy[5] = -ye
    dx[6] = -(adot*t_emit*t_emit - g*xe*t_emit)
    dy[6] = -(bdot*t_emit*t_emit - g*ye*t_emit)

    return x,y,dx,dy


def prelim_fit(obsarray,pout):
    """Take a set of observations and make a preliminary fit using the
     linear model of the orbit.  Then fill in zero for gdot, and return
     the fit and an uncertainty matrix.  The gdot term of uncertainty
     matrix is set to a nominal value.
     Note covar is assumed to be 6x6 1-indexed matrix a la Numerical Recipes.
    """
    beta=np.zeros(6)
    alpha=np.zeros((6,6))

    # Collect the requisite sums
    for ob in obsarray:
        # not sure why the original had this? if(obsarray[i].reject==0)

        wtx = 1./ob.dthetax
        wtx *= wtx
        wty = 1./ob.dthetay
        wty *= wty

        x,y,dx,dy = kbo2d_linear(pout,ob,t_ref);
        """Note that the dx[6] and dy[6] terms will only make
        even the least sense if the g and adot were set to
        some sensible value beforehand.
        """

        for j in range(6):
          beta[j] += ob.thetax * dx[j] * wtx;
          beta[j] += ob.thetay * dy[j] * wty;
          for k in range(j+1):
              alpha[j][k] += dx[j]*dx[k]*wtx;
              alpha[j][k] += dy[j]*dy[k]*wty;

    """ Symmetrize and invert the alpha matrix to give covar.  Note
    that I'm only going to bother with the first 5 params.
    """
    for i in range(5):
      for j in range(i):
          alpha[j][i]=alpha[i][j];

    covar = np.linalg.inv(alpha)
    soln = np.dot(covar, beta)

    pres = np.append(soln,0.0)

    # Set the gdot parts of the covariance matrix to nominal values
    for i in range(6):
        covar[i][5]=covar[5][i]=0.;
    covar[5][5]=0.1*TPI*TPI*g**3

    return pres, covar


def chisq(opt_result,args):
    """ calc the chi sq for the result of the minimization function.
    NOTE: probably wrong... """
    chi,var_chi=0.0,(0.2/205625.)**2 # in radians (from arcseconds)
    a,adot,b,bdot,g,gdot = opt_result
    chi_arr = []
    for arg in args:
        xe, ye, ze, t_emit, theta_x, theta_y = arg
        tx = (a + adot*t_emit - g*xe)/(1 + gdot*t_emit - g*ze)
        ty = (b + bdot*t_emit - g*ye)/(1 + gdot*t_emit - g*ze)
        chi += (theta_x-tx)**2/var_chi + (theta_y-ty)**2/var_chi
        chi_arr.append((theta_x-tx)**2/var_chi + (theta_y-ty)**2/var_chi)

    return chi,chi_arr

def vis_arrows_by_chi(params_dict,chi_res):
    arrows = []
    for v,ch in zip(params_dict.values(),chi_res):
        # k represents cluster id and v represents the related a adot,b,bdot,g,g_dot
        arrows.append(list(v[:4])+[np.log(ch)])
    a,adot,b,bdot,colors = np.split(np.array(arrows),5,axis=1)

    # colormap = mlc.ListedColormap (['grey','blue','red'])
    colormap = cm.viridis
    fig,ax=plt.subplots(figsize=(18, 16))

    Q = ax.quiver(a, b, adot, bdot, colors, cmap=colormap,scale=0.3, width=0.0003)
    # ax.quiverkey(Q, X, Y, U, label,)
    plt.colorbar(Q,ax=ax)
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.title('Arrows (just cluster centers) colored by log transform of the chi sq value')
    plt.savefig('arrows_with_chisq_color.pdf')
    # plt.show()

def vis_arrows_by_chi_with_trkl(params_dict, agg_dict, idxs, t_ref, chi_res, g_init=0.4, gdot_init=0.0):
        a,adot,b,bdot,colors = [],[],[],[],[]
        arrows,g_gdots = [],[]
        cluster_tracklet_level = []

        for k,v in agg_dict.items():
            if k in idxs:
                cluster_tracklet_level.append(v)

        if len(params_dict)!= len(cluster_tracklet_level):
            raise ValueError('the length of the cluster params {0} is different from the number of ids \
                                passed for agg_dict {1}'.format(len(params_dict),len(cluster_tracklet_level)))
        for k,v,ch in zip(params_dict.keys(),params_dict.values(),chi_res):
            # k represents cluster id and v represents the related a adot,b,bdot,g,g_dot
            arrows.append(list(v[:4])+[np.log(ch)])
            g_gdots.append((k,v[4:]))
        # for clust_trkls,cparams in zip(cluster_tracklet_level,params_dict.values()):
        #     g_cl,gdot_cl = cparams[-2:]
        #     for trkl in clust_trkls:
        #         obs_in_trkl = [i[1:] for i in trkl]
        #         arrows.append(list(fit_tracklet(t_ref, g_cl, gdot_cl, obs_in_trkl)[:4])+[1])
        cl_trk_arrows = []
        for clust_trkls in cluster_tracklet_level:
            for trkl in clust_trkls:
                obs_in_trkl = [i[1:] for i in trkl]
                cl_trk_arrows.append(list(fit_tracklet(t_ref, g_init, gdot_init, obs_in_trkl)[:4])+[1])
        # print(arrows)
        # print([a for a in arrows if a[-1]==10])
        ac,adotc,bc,bdotc,colorsc = np.split(np.array(cl_trk_arrows),5,axis=1)
        a,adot,b,bdot,colors = np.split(np.array(arrows),5,axis=1)

        # colormap = mlc.ListedColormap (['grey','blue','red'])
        colormap = cm.cool
        fig,ax=plt.subplots(figsize=(18, 16))

        Q = ax.quiver(a, b, adot, bdot, colors, cmap=colormap,scale=0.3, width=0.0003)
        ax.quiver(ac,bc,adotc,bdotc,colorsc,cmap=cm.gray, scale=0.3, width=0.0003, alpha=0.5)
        for pa,pb,pad,pbd,ggd in zip(a, b, adot, bdot,g_gdots):
            # label='g: {0:.6f}, gdot: {0:.6f}'.format(ggd[1][0],ggd[1][1])
            # label='id:%s'%ggd[0]
            if -0.1<pa[0]<0.1 and -0.1<pb[0]<0.1:
                plt.quiverkey(Q, X=pa[0], Y=pb[0], U=pad[0],label='g: {0:.6f}, gdot: {1:.6f}'.format(ggd[1][0],ggd[1][1]), coordinates='data')
        plt.colorbar(Q,ax=ax)
        plt.xlim(-0.1, 0.1)
        plt.ylim(-0.1, 0.1)
        plt.xlabel('alpha')
        plt.ylabel('beta')
        plt.title('Arrows colored by the log transform of the chi squared value. tracklets are overlayed in light grey')
        plt.savefig('arrows_with_chi_and_trkl.pdf')
        # plt.show()
        return g_gdots

def visualize(params_dict, agg_dict, idxs, t_ref, g_init=0.4, gdot_init=0.0):
    """ This function displays and saves a quiver plot with the arrows form our
    fitted clusters (passed in params_dict), and our original measures passed in
    agg_dict (only the netries with the realted cluster id's passed in idxs)."""
    a,adot,b,bdot,colors = [],[],[],[],[]
    arrows = []
    cluster_tracklet_level = []

    for k,v in agg_dict.items():
        if k in idxs:
            cluster_tracklet_level.append(v)

    if len(params_dict)!= len(cluster_tracklet_level):
        raise ValueError('the length of the cluster params {0} is different from the number of ids \
                            passed for agg_dict {1}'.format(len(params_dict),len(cluster_tracklet_level)))
    for k,v in params_dict.items():
        # k represents cluster id and v represents the related a adot,b,bdot,g,g_dot
        arrows.append(list(v[:4])+[1000])

    for clust_trkls,cparams in zip(cluster_tracklet_level,params_dict.values()):
        g_cl,gdot_cl = cparams[-2:]
        for trkl in clust_trkls:
            obs_in_trkl = [i[1:] for i in trkl]
            arrows.append(list(fit_tracklet(t_ref, g_cl, gdot_cl, obs_in_trkl)[:4])+[500])

    for clust_trkls in cluster_tracklet_level:
        for trkl in clust_trkls:
            obs_in_trkl = [i[1:] for i in trkl]
            arrows.append(list(fit_tracklet(t_ref, g_init, gdot_init, obs_in_trkl)[:4])+[100])
    # print(arrows)
    # print([a for a in arrows if a[-1]==10])
    a,adot,b,bdot,colors = np.split(np.array(arrows),5,axis=1)

    # colormap = mlc.ListedColormap (['grey','blue','red'])
    colormap = cm.cool
    fig,ax=plt.subplots(figsize=(18, 16))

    Q = ax.quiver(a, b, adot, bdot, colors, cmap=colormap,scale=0.3, width=0.0003)

    plt.colorbar(Q,ax=ax)
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.title('Arrows with orbit fit. Darker colors represent fitted clusters, and lighter clusters represent individual tracklets.')
    plt.savefig('arrows_with_orb_fit.pdf')
    # plt.show()


def full_fit_t_loss(t_ref, g_init, gdot_init,  list_of_tracklets, GM=MPC_library.Constants.GMsun):
    """ This function needs to take in all the observations over the cluster of
    tracklets (min of 3 tracklets), and return the a,adot,b,bdot,g and gdot.

    We will then use the resulting gamma and gamma dot to fit the tracklets
    individualally, and compare with the chi sq.
    --------
    Args: t_ref; lunation_center
          g_init; out initial guess for g (the value we asserted before)
          gdot_init; out initial guess for gdot (the value we asserted before)
          all_obs; list of tuples, where each tuple is a observation and all the
                    observations make up at minimum 3 tracklets in a cluster.
    --------
    Returns: tuple with (parms, function min, chisq) where the first element is
                parameters calculated by the nonlinear fit, the second is the
                value of the loss function when completely minimized, and the third
                value is the chisq statistic.
    """
    # tracklet id, jd, dtl, theta_x, theta_y, theta_z, xe, ye, ze, t_emit
    # dependent = np.array([np.array((obs[2],obs[3])) for obs in all_obs])
    # for c,clust in agg_dict.items():
    working_obs = [itm[1:] for ob in list_of_tracklets for itm in ob]
    # working_obs = [i[1:] for i in a_obs]
    t_ref_mean = sum(obs[0] for obs in working_obs)/len(working_obs)
    # print('Avg reference time:',t_ref_mean)
    args = [(obs[5],obs[6],obs[7],obs[0]-obs[1]-t_ref,obs[2],obs[3]) for obs in working_obs]

    # get the avg of the trackelt params
    x0_guess = []
    for trkl in list_of_tracklets:
        obs_in_trkl = [i[1:] for i in trkl]
        x0_guess.append(np.array(fit_tracklet(t_ref, g_init, gdot_init, obs_in_trkl)[:4]))
    x0_guess = np.append(np.array(x0_guess).mean(axis=0), [g_init,gdot_init])

    def loss(arr):
        """ loss func: aggregate the errors from the loss and minimize this function
        Kind of nasty because it uses global args values, but it should work."""
        loss= 0.0
        a, adot, b, bdot, g, gdot = arr

        for arg in args:
            xe, ye, ze, t_emit, theta_x, theta_y = arg
            tx = (a + adot*t_emit - g*xe)/(1 + gdot*t_emit - g*ze)
            ty = (b + bdot*t_emit - g*ye)/(1 + gdot*t_emit - g*ze)
            loss+= (theta_x-tx)**2 + (theta_y-ty)**2

        return loss

    opt_out = minimize(loss,x0=np.array(x0_guess))#,method='L-BFGS-B')

    # calc chi sq
    csq,csq_arr = chisq(opt_out.x,args)

    # print('init',x0_guess)
    # print('diff',[abs(i-j) for i,j in zip(x0_guess,opt_out.x)])
    # print('everything',opt_out)
    return opt_out.x, opt_out.fun, csq,csq_arr

# results_dict[trackletID].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))
def fit_extend(infilename, clust_ids, pixels, nside, n, angDeg=5.5, gi=0.4, gdoti=0.0):
    """ Not a real this yet.. still in progress wrapper for doing the whole nlin fit process"""
    res_dict = get_res_dict(infilename, pixels, nside, n, angDeg=angDeg, gi=g, gdoti=gdot)

    for i in pixels:
        agg_dict = defaultdict(list)
        for k,v in clust_ids.items():
            # k is the tracklet id, v is the cluster id
            agg_dict[v].append(res_dict[i][k])
        fit_dict, results = nlin_fits(agg_dict,g,gdot,n)
        for k,v in fit_dict.items():
            # k is the cluster id and v is the fitted 6 params
            a,adot,b,bdot,g,gdot = v
            """ I need to match the other values from res_dict with the
            fitted cluster values here."""

def nlin_fits(agg_dict, g_init, gdot_init, n):
    """ helper for nlin fits"""
    fit_dict,results = {},[]
    # k is the cluster id, v is the tracklets in the cluster id
    for k, v in agg_dict.items():
        params, func_val, chisq = full_fit_t_loss(util.lunation_center(n), g, gdot, [obs for trkl in v for obs in trkl])
        results.append(params)
        fit_dict[k] = params
    return fit_dict, results
#     for clust_id, arrows in fit_dict.items():
#
#         i = 0
#         label_dict={}
#         combined=[]
#         for k, cx, mx, cy, my, t in arrows:
#             label_dict[i] = k
#             combined.append([cx, mx*dt, cy, my*dt])
#             i +=1
#         points=np.array(combined)
#
#         tree = scipy.spatial.cKDTree(points)
#         matches = tree.query_ball_tree(tree, rad)
#
#         for j, match in enumerate(matches):
#             if len(match)>=mincount:
#                 cluster_list =[]
#                 for idx in match:
#                     tracklet_id = label_dict[idx].strip()
#                     cluster_list.append(tracklet_id)
#                     cluster_id_dict.update({tracklet_id: j})
#                 cluster_key='|'.join(sorted(cluster_list))
#                 cluster_counter.update({cluster_key: 1})
#
# return cluster_counter, cluster_id_dict

# results_dict[trackletID].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))
def full_fit_trkl(t_ref, g_init, gdot_init,  all_obs, GM=MPC_library.Constants.GMsun):
    """ This is not working... """
    # theta_x, theta_y, theta_z, xe, ye, ze, t_emit
    dependent = np.array([np.array((obs[2],obs[3])) for obs in all_obs])
    independent = np.array([np.array((obs[5],obs[6],obs[7],obs[0]-obs[1]-t_ref)) for obs in all_obs])

    p0_guess = list(fit_tracklet(t_ref, g_init, gdot_init, all_obs)[:4])
    p0_guess.extend([g_init,gdot_init])

    print('init',p0_guess)

    def f(obs,a,adot,b,bdot,g,gdot):
        """ """
        # print('obs',obs.shape)
        # print(np.split(obs,4,axis=1))
        # xe,ye,ze,t_emit = np.split(obs,4,axis=1)
        xe,ye,ze,t_emit = obs
        x = (a + adot*t_emit - g*xe)/(1 + gdot*t_emit - g*ze)
        y = (b + bdot*t_emit - g*ye)/(1 + gdot*t_emit - g*ze)

        # print(x,y)
        # print(dependent.shape)
        # print(res.shape)
        # print(res[:,0].reshape(-1,1))
        # print(res[:,0])
        return np.array([x,y])

    def loss(obs,a,adot,b,bdot,g,gdot):
        """ loss func: aggregate the errors from the loss and minimize this function """
        loss,chi = 0.0,0.0
        var_chi = 0.2/205625. # in radians (conversion from arcseconds)
        # a, adot, b, bdot, g, gdot = arr


        xe, ye, ze, t_emit, theta_x, theta_y = np.split(obs,6,axis=1)
        tx = (a + adot*t_emit - g*xe)/(1 + gdot*t_emit - g*ze)
        ty = (b + bdot*t_emit - g*ye)/(1 + gdot*t_emit - g*ze)
        loss = (theta_x-tx)**2 + (theta_y-ty)**2
        # chi += numerator_chi/var_chi
        # loss += numerator_chi

        return loss

    print(independent.shape,dependent.shape)
    params, pcov = curve_fit(f, xdata=independent, ydata=dependent, p0=p0_guess)
    print('init',p0_guess)
    print('params',params)
    print('cov',pcov)
    print('reduced chi sq?',sum((f(independent,*params)-dependent)**2)/(len(all_obs)-6.0))
    return params

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
        vec = hp.pix2vec(nside, pix, nest=True)
        neighbors = hp.query_disc(nside, vec, angDeg*np.pi/180., inclusive=True, nest=True)
        lines = []

        for pix in neighbors:
            for line in hp_dict[pix]:
                lines.append(line)
        if len(lines) > 0:
            res_dict = get_tracklet_obs(vec,lines)
            master_dict[i] = res_dict

    return master_dict

def chi_sq_compare(t_ref, g, gdot, tracklets,  fit_tracklet_func=fit_tracklet, GM=MPC_library.Constants.GMsun):
    """ FUNCTION NOT DONE OR USED::::: Here we want to take in the previous a,
     adot, b, bdot and compare those
    values via chi sq with the orbit calculated by using all the obs in the
    clustered tracklets.  The input will be the reference time, g, gdot and a
    list of lists where the inner list is the observations in a tracklet
    and the meta list is the list of tracklets in our cluster.
    """
    # Get the fitted a,adot,b,bdot values on a per-tracklet level
    individual_fits = []
    for tracklet in tracklets:
        individual_fits.append(np.array(list(fit_tracklet_func(t_ref, g, gdot, tracklet))[:4]+[g,gdot]))

    # Get the fitted values over all observations in the cluster
    all_obs = [obs for track in tracklets for obs in track]
    meta_fit = np.array(full_fit_trkl(t_ref, g, gdot, all_obs))

    validated_cluster = []
    for tracklet, ifit in zip(tracklets, individual_fits):
        # We can do this because we expect the a,adot,b,bdot to have the same scale, right?
        chi2stat, pval = chisquare(ifit,meta_fit)
        if pval>0.05:
            validated_cluster.append(tracklet)

    # Now return a list of lists of tracklets we validated are correctly clustered.
    # the second return value is a bool stating if all the tracklets we initially had were in the valid cluster
    return validated_cluster, len(validated_cluster)==len(tracklets)


# results_dict[trackletID].append((jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))
def fit_tracklet_tst(t_ref, g, gdot, v, GM=MPC_library.Constants.GMsun):
    """ coupled least squares fit for the parameters holding gamma constant.

    NOTE: gamma dot input is not needed and is left for method consistency

    This is not used anywhere yet, and the purpose is to solve for, linearly,
    what the nlin fit solves for nonlinearly.
    --------
    Args:
          t_ref; float, the lunar center jd
          g; float; the gamma value (distance from observer to the asteroid)
          gdot; float; the gamma dot value of radial velocity.
    --------
    Returns: solution vector with calculated a,adot,b,bdot,gdot, and given g and time
    """
    # get some vals
    t_emit = [(obs[0]-obs[1]-t_ref) for obs in v]
    acc_z = -GM*g*g

    # make the matrices from the given vector inputs
    mx = np.array([np.array((1.0, t_emit, 0.0, 0.0, g*t_emit*obs[5])) for obs in v])
    my = np.array([np.array((0.0, 0.0, 1.0, t_emit, g*t_emit*obs[6])) for obs in v])
    theta_x = np.array([obs[2] for obs in v])
    theta_y = np.array([obs[3] for obs in v])

    # aggregate the matrices for the coupoled lstsq fit
    combined_m = np.dot(mx.T,mx)+np.dot(my.T,my)
    combined_res = np.dot(mx,theta_x) + np.dot(my,theta_y)

    # solution in form: a adot, b, bdot, g_dot
    sol = np.linalg.solve(combined_m, combined_res)

    return (sol, t_emit[0])

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

            # Append the resulting data to a dictionary keye do trackletID.
            # 12/11 added strip here, could propagate errors forward
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

def write_transform_to_arrows(t_ref, g_gdot_pairs, vec, lines, outfilename, cluster_id_dict={}, fit_tracklet_func=fit_tracklet):
    """ wrapper function for _write_arrows_files"""
    results_dict = get_tracklet_obs(vec,lines)
    _write_arrows_files(results_dict, t_ref, g_gdot_pairs, fit_tracklet_func, vec, outfilename,cluster_id_dict)


def transform_to_arrows(t_ref, g_gdot_pairs, vec, lines, outfilename='', fit_tracklet_func=fit_tracklet):
    # TODO put this method together with the one above it and edit where needed.
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
    # This bit from here
    hp_dict = util.get_hpdict(infilename)

    pixel_results = {}
    print('Starting run...')
    for prog,i in enumerate(pixels):
        # Percent complete
        out = prog * 1. / len(pixels) * 100
        sys.stdout.write("\r%d%%" % out)
        sys.stdout.flush()

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

    sys.stdout.write("\r%d%%" % 100)
    print('\n')
    print('Run finished!')
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
    cluster_id_dict = {}
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
                        cluster_id_dict.update({tracklet_id: j})
                    cluster_key='|'.join(sorted(cluster_list))
                    cluster_counter.update({cluster_key: 1})

    return cluster_counter, cluster_id_dict


def _rates_to_results(rates_dict, dts):
    """ helper for train clusters, used in the training method, NA to orb fit"""
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

gs =[0.4]
gdots = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
g_gdots = [(x,y) for x in gs for y in gdots]

def train_clusters(pixels, infilename, t_ref, g_gdots=g_gdots,
                    dts=np.arange(5, 30, 5), radii=np.arange(0.0001, 0.0100, 0.0001),
                    cluster_sky_function=cluster_sky_regions, mincount=3):
    """ This function performs a training run over a set of training data provided
    by Matt Payne of the MPC in December 2017. Finish docs later as this is NA to orb fit"""
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
    out how well their chosen dt ad rad hyper paremeters perform on training data.
    Again, NA to orb fit. """
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
          mincount; int, the fewest number of allowable trackelts is a cluster.
    ----------
    Returns: the cluster_counter and cluster_id_dict objects (mentioned above)
    """
    master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)

    cluster_counter, cluster_id_dict = _get_cluster_counter(master, dt, rad, mincount)
    return cluster_counter, cluster_id_dict



## TODO why is the n not specified and alwasy -11??? and angle is different
def output_sky_regions(pixels, infilename, nside=8, n=-11, angDeg=7.5):
    """ Just gets the lines of a file for the pixels specified.
    Not super realted to orb fit, but you could easily use it as a helper."""
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
    """ this counts all the clusters, and returns several counters, not super
    applicable for orb fit"""
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

    clusterID_set = set()
    for pix, cntr in mergedCounter_dict.items():
        for cID, count in cntr.items():
            if count>=mincount:
                clusterID_set.update({cID})

    # return true_counts, mergedCounter_dict, mergedTime_dict
    return clusterID_set, mergedCounter_dict, mergedTime_dict

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
            write_transform_to_arrows(util.lunation_center(n), [(g, gdot)], vec, lines, outfilename, cluster_id_dict=cluster_id_dict)



############################################################################
# def find_clustersOLD(pixels, infilename, t_ref, g_gdots=g_gdots,
#                     rtype='run', dt=15, rad=0.00124,
#                     cluster_sky_function=cluster_sky_regions, mincount=3):
#     """docs"""
#     valid_rtypes = ['run','test','train']
#
#     # Check for valid type input
#     if rtype not in valid_rtypes:
#         raise ValueError('You must use a rtype in {0}, not {1}'.format(valid_rtypes,rtype))
#
#     # Set different defaults for training
#     if dt==15 and rad==0.00124 and rtype=='train':
#         dt, rad = np.arange(5, 30, 5), np.arange(0.0001, 0.0100, 0.0001)
#
#     if (hasattr(dt, '__len__') or hasattr(rad, '__len__')) and rtype!='train':
#         raise ValueError('test and run rtypes can only take scalar dt and rad values')
#
#     if not hasattr(rad, '__len__') and rtype=='train':
#         raise ValueError('train rtypes can only take a list or array of dt and rad values')
#
#     master = cluster_sky_function(g_gdots, pixels, t_ref, infilename)
#
#     if rtype=='run':
#         cluster_counter, cluster_id_dict = _get_cluster_counter(master, dt, rad, mincount)
#         return cluster_counter, cluster_id_dict
#
#     elif rtype=='test':
#         cluster_counter, cluster_id_dict = _get_cluster_counter(master, dt, rad, mincount)
#         test_set = list(cluster_counter.keys())
#         success_dict, failure_counter = unique_clusters(test_set)
#         return len(success_dict), len(failure_counter), list(success_dict.keys()), list(failure_counter.keys())
#     else:
#         # The training case
#         rates_dict={}
#         for dt_val in dt:
#             for rad_val in rad:
#                 cluster_counter, cluster_id_dict = _get_cluster_counter(master, dt_val, rad_val, mincount)
#                 # This region from here
#                 errs = 0
#                 for i, k in enumerate(cluster_counter.keys()):
#                     keys = k.split('|')
#                     stems = [key.split('_')[0] for key in keys]
#                     stem_counter = Counter(stems)
#                     if len(stem_counter)>1:
#                         errs +=1
#
#                 rates_dict[dt_val, rad_val] = cluster_counter.keys(), errs
#
#         return _rates_to_results(rates_dict, dt)
