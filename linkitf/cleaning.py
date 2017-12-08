# Imports
import scipy.interpolate
import numpy as np
import pandas as pd
import math
import healpy as hp
import collections
import astropy
from collections import defaultdict
from lib import MPC_library
import util


def is_two_line(line):
    """ This routine checks the 80-character input line to see if it contains
    a special character (S, R, or V) that indicates a 2-line record. """
    note2 = line[14]
    return note2=='S' or note2=='R' or note2=='V'


def split_MPC_file(filename):
    """ This routine opens and reads filename, separating the records into
    those in the 1-line and 2-line formats. The 2-line format lines are merged
    into single 160-character records for processing line-by-line. """
    filename_1_line = filename.rstrip('.txt')+"_1_line.txt"
    filename_2_line = filename.rstrip('.txt')+"_2_line.txt"
    with open(filename_1_line, 'w') as f1_out, open(filename_2_line, 'w') as f2_out:
        line1=None
        with open(filename, 'r') as f:
            for line in f:
                if is_two_line(line):
                    line1=line
                    continue
                if line1 != None:
                    merged_lines = line1.rstrip('\n') + line
                    f2_out.write(merged_lines)
                    line1 = None
                else:
                    f1_out.write(line)
                    line1 = None

def convertObs80(line):
    """ Special case conversion for a corrupted file line in the ITF file."""
    objName   = line[0:5]
    provDesig = line[5:12]
    disAst    = line[12:13]
    note1     = line[13:14]
    note2     = line[14:15]
    dateObs   = line[15:32]
    RA        = line[32:44]
    Dec       = line[44:56]
    mag       = line[65:70]
    filt      = line[70:71]
    obsCode   = line[77:80]
    return objName, provDesig, disAst, note1, note2, dateObs, RA, Dec, mag, filt, obsCode


def get_sorted_tracklets(itf_filename):
    """ From the MPC file pull out all the tracklets."""
    tracklets = defaultdict(list)
    tracklets_jd_dict = {}
    with open(itf_filename) as infile:
        for line in infile:
            if not line.startswith('#'):
                desig = line[0:12]
                jd_tdb = float(line[43:57])
                if desig not in tracklets_jd_dict:
                    # Got a new tracklet
                    tracklets_jd_dict[desig] = jd_tdb
                tracklets[desig].append(line)
    sortedTrackletKeys = sorted(tracklets.keys(), key=lambda k: tracklets_jd_dict[k])
    return tracklets, tracklets_jd_dict, sortedTrackletKeys


def separate_time_windows(tracklets, sortedTracklets, tracklets_jd_dict, file_stem, \
                                    n_begin=-825, n_end=14, dt=15., suff='.mpc'):
    """ Sweep through the tracklets once, outputting them into a sequence of
    overlapping time ranges that can be processed separately.
    -------
    Args: tracklets; dict, the first item returned from get_sorted_tracklets.
          sortedTracklets; list, the second item returned from get_sorted_tracklets.
          tracklets_jd_dict; dict, the third item returned from get_sorted_tracklets.
          file_stem; str, path to the realted mpc file that we are splitting up.
          n_begin; int, the beginning index for the lunar centers.
          n_end; int, the end index for the lunar centers.
          dt; float, the day scale factor used to weight realtive importance of
                position and velocity.
          suff; str, the suffix of the file, normally ".mpc" but sometimes ".txt"
    --------
    Returns: None; it just writes files to the directory you specify.
    """
    t_center = util.lunation_center(n_begin)
    files = {}

    header='#trackletID yr   mn dy      obsCode mag filter  jd_tdb       x_target     y_target     z_target      x_obs       y_obs        z_obs     '

    for desig in sortedTracklets:
        jd_tdb = tracklets_jd_dict[desig]
        while(jd_tdb>t_center+dt):
            if n_begin in files:
                files[n_begin].close()
            n_begin +=1
            t_center = util.lunation_center(n_begin)
        for n in range(n_begin, n_end):
            if jd_tdb<util.lunation_center(n)-dt:
                break
            if n not in files:
                outfile = file_stem.rstrip('.mpc')+'_'+str(util.lunation_center(n))+'_pm'+str(dt)+suff
                files[n] = open(outfile, 'w')
                files[n].write(header+'\n')
            for line in tracklets[desig]:
                files[n].write(line)


def adjust_position(r, rho_target, re):
    """ This returns the topocentric distances and new heliocentric
    position vectors to the target, given the assumed distance
    r and the position vector of the observatory re."""
    rho_x, rho_y, rho_z = rho_target
    xe, ye, ze = re
    Robs = np.sqrt(xe * xe + ye * ye + ze * ze)
    cos_phi = -(rho_x * xe + rho_y * ye + rho_z * ze) / Robs
    phi = np.arccos(cos_phi)
    sin_phi = np.sin(phi)

    xx2 = r*r - Robs*sin_phi * Robs*sin_phi

    if xx2 < 0:
        None, None

    xx = np.sqrt(xx2)
    yy = Robs * cos_phi

    rho_p = yy + xx

    # This could be done with numpy arrays
    x_p = xe + rho_p*rho_x
    y_p = ye + rho_p*rho_y
    z_p = ze + rho_p*rho_z

    rho_m = yy - xx

    # This could be done with numpy arrays
    x_m = xe + rho_m*rho_x
    y_m = ye + rho_m*rho_y
    z_m = ze + rho_m*rho_z

    return (rho_p, (x_p, y_p, z_p)), (rho_m, (x_m, y_m, z_m))

def index_positions(n, r_func, file_stem, dt=45., nside=8):
    """
    Does the transformations on the data using the date of the n-th new
    moon as the reference time.

    It is reading and processing the entire *.mpc file.

    This does the heliocentric tranformation for the assumed radius function,
    r_func.

    It then does light-time correction.

    And it appends a healpix number on each line in order to be able to quickly
    select data from a given region of sky.

    This generates a file called *.trans, and it incorporates
    the distance assumed in the file name.
    """
    infilename = file_stem.rstrip('.mpc')+'_'+str(util.lunation_center(n))+'_pm'+str(dt)+'.mpc'
    try:
      open(infilename, 'r')
    except IOError:
      return 0
    t_ref = lunation_center(n)
    r_ref = r_func(t_ref)
    r_name = "_r%.1lf" % (r_ref)
    outfilename = file_stem.rstrip('.mpc')+'_'+str(util.lunation_center(n))+'_pm'+str(dt)+r_name+'.trans'

    with open(infilename, 'r') as infile, open(outfilename, 'w') as outfile:
        for line in infile:
            if line.startswith('#'):
                header = line.rstrip()
                outfile.write(header + '          dt         x_cor       y_cor        z_cor       pix \n')
            else:
                lineID = line[:43]

                jd_tdb = float(line[43:57])

                x_target, y_target, z_target = line[57:97].split()
                r_target = np.array([float(x_target), float(y_target), float(z_target)])

                x_obs, y_obs, z_obs = line[97:135].split()
                r_obs = np.array([float(x_obs), float(y_obs), float(z_obs)])

                # This should be a function from here
                # Adjust positions
                dt = 0.0
                r_prev = r_func(jd_tdb-dt)
                rho_r_p, rho_r_m = adjust_position(r_prev, r_target, r_obs)
                dt = rho_r_p[0]/MPC_library.Constants.speed_of_light

                # Do light-time iterations.
                # Probably don't need to do this at this point, because it is
                # being re-done in a later step.
                i=0
                while(np.abs(r_func(jd_tdb-dt)-r_prev)>1e-8):
                    rho_r_p, rho_r_m = adjust_position(r_prev, r_target, r_obs)
                    dt = rho_r_p[0]/MPC_library.Constants.speed_of_light
                    r_prev = r_func(jd_tdb-dt)
                    i += 1

                # to here
                xp, yp, zp = rho_r_p[1]

                # Calculate HEALPix index
                pix = hp.vec2pix(nside, xp, yp, zp, nest=True)

                outstring = line.rstrip() + " %13.6lf %12.7lf %12.7lf %12.7lf %5d\n"% \
                      (dt, xp, yp, zp, pix)

                outfile.write(outstring)
