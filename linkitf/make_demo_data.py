# Imports
from lib import MPC_library
import cleaning as cl
import scipy.interpolate

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
