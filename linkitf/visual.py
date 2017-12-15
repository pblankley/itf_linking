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
import scipy.spatial
import pickle
from operator import add
import matplotlib.cm as cm
import matplotlib.colors as mlc
from mpl_toolkits.mplot3d import Axes3D


def number_clusters_plot(pix_runs,true_count,outpath=''):
    """ This function plots the number of clusters we find vs the
    cluster radius we define.  It is used as a visual for training.
    -------
    Args: pix_runs; the result of looping over various healpy pixel values
            in training and putting them in a dictionary.  To learn more
            about this data structure look at the docs for tuning.
          true_count; int, the true number of clusters in training.
            Get this value by summing up the values of true_count_dict from
            "accessible_clusters.
          outpath; str, where you want the file to go (e.g. 'plots/arrows')
    -------
    Returns: None; plots the graph, and saves the fig
    """
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
    plt.legend()
    plt.title('Number of clusters by cluster radius.')
    if outpath != '':
        plt.savefig(outpath)
    plt.show()


def number_errors_plot(pix_runs,outpath=''):
    """ This function plots the number of errors we have vs the
    cluster radius we define.  It is used as a visual for training.
    -------
    Args: pix_runs; the result of looping over various healpy pixel values
            in training and putting them in a dictionary.  To learn more
            about this data structure look at the docs for tuning.
          outpath; str, where you want the file to go (e.g. 'plots/arrows')
    -------
    Returns: None; plots the graph, and saves the fig
    """
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

        plt.plot(ds, nerrors, label=dt)

    plt.xscale("log", nonposx='clip')
    plt.ylim((0,3000))
    plt.xlabel('d (cluster radius)')
    plt.ylabel('N errors')
    plt.text(0.0005, 1000, r'$\gamma=0.4$', fontsize=15)
    plt.legend()
    plt.title('Number of errors by cluster radius')
    if outpath != '':
        plt.savefig(outpath)
    plt.show()


def auc_plot(pix_runs,true_count,outpath=''):
    """ This function plots the number of errors we have as a percentage
    of the total true clusters vs the number of clusters we find as a
    percentage of the total true clusters. This approximates the AUC measure.
    It is used as a visual for training.
    -------
    Args: pix_runs; the result of looping over various healpy pixel values
            in training and putting them in a dictionary.  To learn more
            about this data structure look at the docs for tuning.
          true_count; int, the true number of clusters in training.
            Get this value by summing up the values of true_count_dict from
            "accessible_clusters.
          outpath; str, where you want the file to go (e.g. 'plots/arrows')
    -------
    Returns: None; plots the graph and saves the fig
    """
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
    plt.title('AUC-proxy plot')
    if outpath != '':
        plt.savefig(outpath)
    plt.show()

def _get_vals_for_plot(filename,cluster,subs):
    """ Get the columns and process based on the
    passed flags for the make_figure function."""
    mxs, cxs, mys, cys, dts =[], [], [], [], []
    # print(cluster,subs)
    if cluster and subs:
        # print('in here')
        for line in open(filename):
            if line.startswith('#'):
                continue
            desig, cx, mx, cy, my, cid = line.split()
            if int(cid)!=-1:

                mxs.append(float(mx))
                cxs.append(float(cx))
                mys.append(float(my))
                cys.append(float(cy))
                dts.append(float(cid))
    else:
        for line in open(filename):
            if line.startswith('#'):
                continue
            desig, cx, mx, cy, my, dt = line.split()
            mxs.append(float(mx))
            cxs.append(float(cx))
            mys.append(float(my))
            cys.append(float(cy))
            dts.append(float(dt))
    return mxs, cxs, mys, cys, dts

def make_figure(filename,cluster=False,subs=True,outpath=''):
    """ This function plots the asteroids on the tangent plane viewed as if
    we are standing on the sun (heliocentric view).  It takes in the file
    produced by generate_sky_region_files and a T/F flag for clustering.
    If you pass in the T flag for clustering, make sure you give it the
    right file.  The file that ends in '_cid' is meant for clustering, and the
    file that has no '_cid' is meant for plotting with color in respect to
    changing time.
    -------
    Args: filename; file produced by generate_sky_region_files.
          cluster; bool, True means color categorically by cluster id
                    False meant color by time.
          outpath; str, where you want the file to go (e.g. 'plots/arrows')
    -------
    Returns: None; plots the graph and saves the fig
    """
    plt.ioff()
    mxs, cxs, mys, cys, dts = _get_vals_for_plot(filename,cluster,subs)

    fig=plt.figure(figsize=(18, 16))

    colormap = cm.inferno
    if cluster:
        colormap = mlc.ListedColormap ( np.random.rand ( 256,3))

    plt.quiver(cxs, cys, mxs, mys, dts, cmap=colormap,scale=0.3, width=0.0003)

    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    if cluster:
        plt.title('Clusters with 3 or more tracklets in this time slice, colored by cluster.')
    else:
        plt.title('All tracklets in this time slice, colored by time difference.')
    if outpath=='':
        outfile = filename+'.pdf'
    else:
        outfile = outpath +'.pdf'
    plt.savefig(outfile)
    plt.close()
    plt.ion()

def make_figure_sharing(filename,cluster=False,subs=True,outpath=''):
    plt.ioff()
    mxs, cxs, mys, cys, cids =[], [], [], [], []
    smxs, scxs, smys, scys, scids =[], [], [], [], []
    # print(cluster,subs)
    if cluster and subs:
        # print('in here')
        for line in open(filename):
            if line.startswith('#'):
                continue
            desig, cx, mx, cy, my, cid = line.split()

            if int(cid)==-42:
                # print('in here')
                smxs.append(float(mx))
                scxs.append(float(cx))
                smys.append(float(my))
                scys.append(float(cy))
                scids.append(float(cid))

            if int(cid)!=-1:

                mxs.append(float(mx))
                cxs.append(float(cx))
                mys.append(float(my))
                cys.append(float(cy))
                cids.append(float(cid))

    fig,ax=plt.subplots(figsize=(18, 16))

    colormap = cm.inferno
    if cluster:
        colormap = mlc.ListedColormap ( np.random.rand ( 256,3))

    ax.quiver(cxs, cys, mxs, mys, cids, cmap=colormap,scale=0.3, width=0.0003)
    ax.quiver(scxs, scys, smxs, smys, color='black',scale=0.3, width=0.0003)

    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    if cluster:
        ax.set_title('Clusters with 3 or more tracklets in this time slice, colored by cluster.')
    else:
        ax.set_title('All tracklets in this time slice, colored by time difference.')
    if outpath=='':
        outfile = filename+'.pdf'
    else:
        outfile = outpath +'.pdf'
    plt.savefig(outfile)
    plt.close()
    plt.ion()

def vis_cluster_arrows_err(params_dict,c_res,log=False,subdir=''):
    """ This function plots arrows (just cluster fits).  The color is
    given by the error value (currently one of chi squared, or rms) for each
    cluster fit.  The plot is the typical quiver plot we show.  The lighter yellow
    (on viridis) represent the larger errors and the darker purple represent the
    low errors.
    ----------
    Args: params_dict;
    """
    arrows = []
    for v,ch in zip(params_dict.values(),c_res):
        # v represents the related a adot,b,bdot,g,g_dot
        if log:
            arrows.append(list(v[:4])+[np.log(ch)])
        else:
            arrows.append(list(v[:4])+[ch])
    a,adot,b,bdot,colors = np.split(np.array(arrows),5,axis=1)

    colormap = cm.viridis
    fig,ax=plt.subplots(figsize=(18, 16))

    Q = ax.quiver(a, b, adot, bdot, colors, cmap=colormap,scale=0.3, width=0.0003)
    plt.colorbar(Q,ax=ax)

    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    if log:
        plt.title('Arrows (just cluster centers) colored by the log error value')
    else:
        plt.title('Arrows (just cluster centers) colored by the error value')
    plt.savefig('{}cluster_arrows_color.pdf'.format(subdir))

def vis_cluster_tracklet_arrows(params_dict, agg_dict, idxs, t_ref, c_res, g_init=0.4, \
                                    gdot_init=0.0,label='None',size=6,log=False,subdir=''):
    """ This function plots arrows (both the cluster fits and the original tracklets).  The color is
    given by the log transform of the chi_sq value for each cluster fit (the tracklets are all grey).
    the label tag tells the function which labels to display. options are ['None','ggdot','id','error']
    """
    valid_labels = ['None','ggdot','id','error']
    if label not in valid_labels:
        raise ValueError('put in a valid lablel. one of {}'.format(valid_labels))

    a,adot,b,bdot,colors = [],[],[],[],[]
    arrows,g_gdots = [],[]
    cluster_tracklet_level = []

    for k,v in agg_dict.items():
        if k in idxs:
            cluster_tracklet_level.append(v)

    if len(params_dict)!= len(cluster_tracklet_level):
        raise ValueError('the length of the cluster params {0} is different from the number of ids \
                            passed for agg_dict {1}'.format(len(params_dict),len(cluster_tracklet_level)))

    for k,v,ch in zip(params_dict.keys(),params_dict.values(),c_res):
        # k represents cluster id and v represents the related a adot,b,bdot,g,g_dot
        if log:
            arrows.append(list(v[:4])+[np.log(ch)])
        else:
            arrows.append(list(v[:4])+[ch])
        g_gdots.append((k,v[4:]))

    cl_trk_arrows = []
    for clust_trkls in cluster_tracklet_level:
        for trkl in clust_trkls:
            obs_in_trkl = [i[1:] for i in trkl]
            cl_trk_arrows.append(list(fit_tracklet(t_ref, g_init, gdot_init, obs_in_trkl)[:4])+[1])

    ac,adotc,bc,bdotc,colorsc = np.split(np.array(cl_trk_arrows),5,axis=1)
    a,adot,b,bdot,colors = np.split(np.array(arrows),5,axis=1)

    colormap = cm.viridis
    fig,ax=plt.subplots(figsize=(18, 16))

    Q = ax.quiver(a, b, adot, bdot, colors, cmap=colormap,scale=0.3, width=0.0003)
    ax.quiver(ac,bc,adotc,bdotc, scale=0.3, width=0.0003, alpha=0.3)

    if label!='None':
        for pa,pb,pad,pbd,ggd,ch in zip(a, b, adot, bdot, g_gdots, c_res):
            if label=='ggdot':
                lab = 'g: {0:.6f}, gdot: {1:.6f}'.format(ggd[1][0],ggd[1][1])
            if label=='id':
                lab = 'id:%s'%ggd[0]
            if label=='error':
                lab = 'error: {}'.format(ch)
            if -0.2<pa[0]<0.2 and -0.2<pb[0]<0.2:
                plt.quiverkey(Q, X=pa[0], Y=pb[0], U=pad[0],label=lab, coordinates='data',fontproperties={'size': size})

    plt.colorbar(Q,ax=ax)
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    if log:
        plt.title('Arrows colored by the log transform of the error values. Tracklets are overlayed in light grey')
    else:
        plt.title('Arrows colored by the error values. Tracklets are overlayed in light grey')
    if log:
        plt.savefig('{0}cluster_tracklet_arrows_label_{1}_log.pdf'.format(subdir,label))
    else:
        plt.savefig('{0}cluster_tracklet_arrows_label_{1}.pdf'.format(subdir,label))
    # return g_gdots

def vis_cluster_tracklet_diff(params_dict, agg_dict, idxs, t_ref, g_init=0.4, gdot_init=0.0,subdir=''):
    """ This function displays and saves a quiver plot with the arrows form our
    fitted clusters (passed in params_dict), and our original measures passed in
    agg_dict (only the entries with the realted cluster id's passed in idxs).
    ============
    The plot is colored by the category of the arrow (cluster fits are dark,
    transformed clusters, with the g and gdot from the cluster fit are inbetween,
    and the original tracklet fits are light blue.)
    """
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

    a,adot,b,bdot,colors = np.split(np.array(arrows),5,axis=1)

    colormap = cm.cool
    fig,ax=plt.subplots(figsize=(18, 16))

    Q = ax.quiver(a, b, adot, bdot, colors, cmap=colormap,scale=0.3, width=0.0003)

    plt.colorbar(Q,ax=ax)
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.title('Arrows with orbit fit. Darker colors represent fitted clusters, and lighter clusters represent individual tracklets.')
    plt.savefig('{}cluster_tracklet_diff.pdf'.format(subdir))

def orbitalElements2Cartesian(a, e, I, peri, node, E):
    """ Convert orbital elements to Cartesian coordinates in the Solar System.
    Args:
        a (float): semi-major axis (AU)
        e (float): eccentricity
        I (float): inclination (radians)
        peri (float): longitude of perihelion (radians)
        node (float): longitude of ascending node (radians)
        E (float): eccentric anomaly (radians)
    """
    # Check if the orbit is parabolic or hyperbolic
    if e >=1:
        e = 0.99999999

    # True anomaly
    theta = 2*np.arctan(np.sqrt((1.0 + e)/(1.0 - e))*np.tan(E/2.0))

    # Distance from the Sun to the poin on orbit
    r = a*(1.0 - e*np.cos(E))

    # Cartesian coordinates
    x = r*(np.cos(node)*np.cos(peri + theta) - np.sin(node)*np.sin(peri + theta)*np.cos(I))
    y = r*(np.sin(node)*np.cos(peri + theta) + np.cos(node)*np.sin(peri + theta)*np.cos(I))
    z = r*np.sin(peri + theta)*np.sin(I)

    return x, y, z

def vis_orbits(orb_elements,limits=True,sun=True,alpha=0.1,c='blue',figsize=(6,6)):
    """ This function is for visualization of orbits fit by our fitting function,
    and it is set up to use "orbital elements" as input. The orbital elements are
    a (semi-major axis), e (eccentricity), i (inclination), w (argument of perigee)
    om (right ascention), m (true/mean anomoly).  They are passed into the fuction
    in the order (a,e,i,w,om,m) in params. v is the only parameter that will vary
    quickly among orbits that are truly similar.

    Source: https://github.com/CroatianMeteorNetwork/CMN-codes/blob/master/Orbit%20Plotter/PlotOrbits.py
    ----------
    Args: params; array or arrays, each inner array is of form (a,e,i,w,om,m)
                    for an orbit as mentioned above.
    ----------
    Returns: None, plots and saves the orbit.
    """
    # Setup the plot
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')

    # plot the Sun
    if sun:
        ax.scatter(0, 0, 0, c='yellow', s=100)

    # Eccentric anomaly (full range)
    E = np.linspace(-np.pi, np.pi, 100)

    # Plot the given orbits
    for i, orbit in enumerate(orb_elements):
        a, e, I, peri, node = orbit

        # Take extra steps in E if the orbit is very large
        if a > 50:
            E = np.linspace(-np.pi, np.pi, int((a/20.0)*100))

        # Get the orbit in the cartesian space
        x, y, z = orbitalElements2Cartesian(a, e, I, peri, node, E)

        # Plot orbits
        # c='#32CD32'
        ax.plot(x, y, z, c=c,alpha=alpha)

    # Add limits (in AU)
    if limits:
        ax.set_xlim3d(-5,5)
        ax.set_ylim3d(-5,5)
        ax.set_zlim3d(-5,5)

    ax.set_title('Asteroid orbits around the sun.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()
    plt.show()
