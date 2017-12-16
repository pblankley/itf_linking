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
import clustering as cl
from collections import defaultdict
from collections import Counter
from libr import MPC_library
import scipy.spatial
import pickle
from operator import add
import matplotlib.cm as cm
from matplotlib.colors import Normalize

Observatories = MPC_library.Observatories

ObservatoryXYZ = Observatories.ObservatoryXYZ

def lunation_center(n, tref=2457722.0125, p=29.53055):
    """ Returns the jd of new moon, to the nearest half day range(-825,14).
    -------
    Args: n (required) the index on which the new moon inquestion occured.
    -------
    Returns: float, the julian date of the new moon to .5 days accurate.
    """
    t = tref + p*n
    tp = np.floor(t) + 0.5
    return tp

def equatorial_to_ecliptic(v,rot_mat=MPC_library.rotate_matrix(-MPC_library.Constants.ecl)):
    """ Convert equatorial plane x,y,z to eliptic x,y,z
    ----------
    Args: v; numpy array, an array that contains the x,y,z coordinates in equatorial
                coordinates.
    ----------
    Returns: numpy array, the x,y,z values in elliptical coordinates.
    """
    return np.dot(rot_mat,v.reshape(-1,1)).flatten()

def xyz_to_proj_matrix(r_ref):
    """ This routine returns the 3-D rotation matrix for the
    given reference vector.
    ---------
    Args: r_ref; numpy array, the reference vector given by healpix's pix2vec()
                function.
    ---------
    Returns: numpy array, the rotation matrix for the given refertence vector.
    """
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

def iswrong(id_set):
    """ Function that checks a list of id's in the training set and determines
    if the id's are correctly matched our not.
    ------
    Args: list of id's
    ------
    Returns: bool, T if incorrectly clustered, F otherwise
    """
    stem_counter = cl.member_counts('|'.join(list(id_set)))
    return len(stem_counter)>1

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
    """ This function goes to the original .mpc file for the data and
    gets all the tracklet ID's and returns a dictionary with tracklet id as the
    key and a list of line numbers as value (starting at the first line of data).
    --------
    Args: filename; str, path to the core .mpc file for your data
    --------
    Returns: dict; where key is tracklet id and the value is a list of all the
                line numbers that tracklet id occurs at
    '"""
    tracklets = defaultdict(list)
    with open(filename) as infile:
        for i, line in enumerate(infile):
            if not line.startswith('#'):
                desig = line[0:12].strip()
                tracklets[desig].append(i)
    return tracklets


def get_original_observation_array(filename):
    """ This function goes to the original .txt file for the data and returns
    the .readlines() result of the entire original file.
    --------
    Args: filename; str, path to the core .txt file for your data
    --------
    Returns: the .readlines() result of the original .txt file.
    '"""
    tracklets = defaultdict(list)
    with open(filename) as infile:
        data = infile.readlines()
    return data

def get_observations(cluster_key, tracklets_dict, observation_array, sep='|'):
    """ This function gets all the original tracklet line information from the
    original .txt file based on the given cluster id (which is a '|' joined
    string of all tracklet id's in the cluster).
    --------
    Args: cluster_key; str, the cluster id where each tracklet id in the cluster
            is joined with the pipe ('|')
          tracklets_dict; dict, with tracklet id as key and index in the original
            file as the value (output of the get_original_tracklets_dict function)
          observation_array; list, of the text lines of the .txt original file
            (output of the get_original_observation_array function).
    --------
    Returns: list of the text lines associated with each tracklet in a cluster.
    """
    array=[]
    for key in cluster_key.split(sep):
        indices = tracklets_dict[key]
        for idx in indices:
            array.append(observation_array[idx].rstrip())
    return array

def get_plot_dict(cid_dict):
    """ The purpose of this function is to be a lightweight way to move from a
    allocated dict to a dict with unique numbers as values (keyed by tracklet id)
    for plotting purposes (pass as the cluster_id_dict argument to the generate_
    sky_region_files).
    ---------
    Args: cid_dict; dict, keyed on tracklet id, with the values as the allocated
            clusters. (Result of the allocate() function).
    ---------
    Returns: cluster_id_dict, a dict keyed on tracklet id with a numeric value for
            cluster_id
    --------
    NOTE: If you pass this function a dict that has not already been allocated,
            it is the same as using the 'coin' method in allocation.
    """
    cluster_id_dict = {}
    ref = {cl: cid for cid,cl in enumerate(sorted(set(cid_dict.values())))}
    for tid,cl in cid_dict.items():
        cluster_id_dict[tid] = ref[cl]
    return cluster_id_dict

def _pbasis_to_State(v):
    """ Takes in the elements a, adot, b, bdot, g, gdot and converts them to
    a inner C object that carries around important information for transformation.
    --------
    Args: numpy array of length 6 with elements a, adot, b, bdot, g, and gdot
    --------
    Returns: State object. useful only for the transformation from elliptic
                coordinates to orbital elements in this context.
    """
    alpha, alpha_dot, beta, beta_dot, gamma, gamma_dot = v
    z = 1.0/gamma
    x = alpha*z
    y = beta*z
    xd = alpha_dot*z
    yd = beta_dot*z
    zd = gamma_dot*z
    return kc.State(x, y, z, xd, yd, zd)


def pbasis_to_elements(pbasis, r_ref, GM=MPC_library.Constants.GMsun):
    """ This function takes in the pbasis array, that is the values
    a, adot, b, bdot, g, gdot.  It then uses the kepcart library to convert
    the pbasis to the orbital elements.  The orbital elements are returned in
    the order a, e, i, big_omega, little_omega, m. Where a is alpha, e is
    eccentricity, i is inclination (angle from the equatorial plane), big_omega
    is the angle from the earth's equatorical reference, little_omega is the angle
    from the line of nodes (intersection of equatorial and orbital planes) to the
    plane on the orbit that is closest to the sun, and m is the mean anomoly (which
    is the angle at which the object is at (located) on its orbit).  The m value is
    the fast varying parameter over time.
    ---------
    Args: pbasis, a numpy array with the values: a, adot, b, bdot, g, gdot
          GM; the universal gravity constant times the center object mass in question
                defaults to G*(mass of sun) as that is our usual orbit center.
    ---------
    Returns: numpy array with the values: a, e, i, big_omega, little_omega, m
    """
    mat = xyz_to_proj_matrix(r_ref).T
    state = _pbasis_to_State(pbasis)
    x, y, z, xd, yd, zd = state.x, state.y, state.z, state.xd, state.yd, state.zd
    xp, yp, zp = np.dot(mat, np.array((x, y, z)))
    xdp, ydp, zdp =  np.dot(mat, np.array((xd, yd, zd)))
    statep = kc.State(xp, yp, zp, xdp, ydp, zdp)
    return kc.keplerian(GM, statep)


def check_dups(clusts):
    """ This function takes in a list of cluster ids (strings) and checks for
    tracklets that are used in mutiple clusters.  If there are clusters that
    share one or more tracklets, this function will return the number of those
    clusters.  If it returns 0 there are no duplicates (shared tracklets) in
    your clusters.
    --------
    Args: clusts; list, string cluster id's
    --------
    Returns: int, the number of clusters that contain a shared tracklet.
    """
    all_clusts = set(clusts)
    trkls,dups = [],0
    for clust in all_clusts:
        if clust=='abstain':
            continue
        flag=0
        for tr in clust.split('|'):
            if tr not in trkls:
                trkls.append(tr)
            else:
                flag=1
        dups+=flag
    return dups

def allocate(final_dict,method='fit',details=False):
    """ This function allocates tracklets to clusters. Since with a KD Tree it is
    possible to have clusters that contain the same tracklet, this step is necessary
    to achieve a definitive result.
    ----------
    Args: final_dict; dict, a dictionary keyed on clusters that are represented with
                the usual '|' joined strings of tracklet ids
          method; str, the method we use for allocation. Must be one of ['fit',
                'smallest','largest','coin']
          details; bool, whether to include the extra details (another dictionary
                keyed on tracklet id with the fitted orbital parameters, and error)
    ----------
    Returns: cld; a dict keyed on tracklet id and the new (potentially smaller) cluster
                id as the value.
             (OPTIONAL) details; a dict keyed on tracklet id with the fitted orbital
                parameters and the error of the fit (stored as a tuple) as the values.
    """
    cld, all_trkls,dets= {}, set(), {}
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
        try:
            all_clusts = list(final_dict.keys())
        except:
            all_clusts = list(final_dict)

    # Allocate
    for i, str_cid in enumerate(all_clusts):
        trkls = set(str_cid.split('|'))
        all_trkls |= trkls
        trkls = sorted(list(trkls))
        tv = [tid in cld.keys() for tid in trkls]
        if not any(tv):
            for tid in trkls:
                cld[tid] = '|'.join(trkls)
                dets[tid] = final_dict[str_cid]
        else:
            if (len(tv)-sum(tv))>=3:
                nids = [tid for tr,tid in zip(tv,trkls) if not tr]
                for tid in nids:
                    cld[tid] = '|'.join(nids)
                    dets[tid] = final_dict[str_cid]

    # Deal with the null (unassigned) tracklets.
    null_trkls = [tid for tid in all_trkls if tid not in cld.keys()]
    for nt in null_trkls:
        cld[nt] = 'abstain'
        dets[nt] = (None,None)
    if details:
        return cld,dets
    return cld



def evalu(cld,details=False):
    """ This function takes in a dictionary keyed on tracklet id and with values
    as cluster id's.  It returns the number of successes, potentials (which we define
    to be tracklets that contain 3 or more of a realted tracklet, but also contain
    at least one "contaminating" tracklet that is not realated.  Failures are where
    we do not have more than 3 correctly grouped tracklets and we have incorrectly
    clustered tracklets in our clusters
    --------
    Args: cld; dict, keyed on tracklet id and values are the clusters.
    --------
    Retuns: int,int,int; corresponding to successes, potentials, and failures
    """
    all_clusts = set(cld.values())
    success,potential,failure,sk,pk,fk = 0,0,0,[],[],[]
    for clust in all_clusts:
        cc = cl.member_counts(clust)
        if len(cc)>1 and any(i>3 for i in cc.values()):
            potential+=1
            pk.append(clust)
        elif len(cc)>1:
            failure+=1
            fk.append(clust)
        else:
            sk.append(clust)
            success+=1
    if details:
        return success,potential,failure,sk,pk,fk
    return success,potential,failure
