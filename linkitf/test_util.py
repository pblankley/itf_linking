import numpy as np
from libr import MPC_library
import util
import numpy as np


def test_lunation():
    assert(util.lunation_center(-11)==2457397.5)

def test_equatorial_to_ecliptic():
    check_ecliptic = np.array((-1.302612238291229, 1.465357398109745, -5.263097826792307e-03))
    equatorial = np.array((-1.302612238291229,1.346532667370993,5.780569003968862e-01))
    assert(np.allclose(util.equatorial_to_ecliptic(equatorial),check_ecliptic,rtol=1e-5,atol=1e-6))

def test_xyz_proj_mat():
    xt,yt,zt = np.array((1,0,0)),np.array((0,1,0)),np.array((0,0,1))
    rm = util.xyz_to_proj_matrix(xt)
    # Test unit vectors
    assert(np.array_equal(np.dot(rm,zt),yt))
    assert(np.array_equal(np.dot(rm,yt),xt))
    assert(np.array_equal(np.dot(rm,xt),zt))

    # Test for rotational property transpose==inverse
    assert(np.array_equal(np.dot(rm,rm.T), np.identity(3)))

    # Test for rotational property magnitude doesnt change
    test1 = np.array((-1.302612238291229, 1.465357398109745, -5.263097826792307e-03))
    rm2 = util.xyz_to_proj_matrix(test1)
    assert(np.allclose(np.dot(rm2,rm2.T),np.identity(3)))
    assert(np.allclose(np.linalg.norm(np.dot(rm2,xt)),np.linalg.norm(xt)))
    assert(np.allclose(np.linalg.norm(np.dot(rm2,test1)),np.linalg.norm(test1)))
    test2= np.array((7.431,1.1132,-2.536465))
    assert(np.allclose(np.linalg.norm(np.dot(rm2,test2)),np.linalg.norm(test2)))

def test_pbasis_to_elem():
    g=0.4
    adot = (2*np.pi*g**1.5)/365.25
    res = util.pbasis_to_elements([0.0, adot, 0, 0.0, g, 0], (1, 0, 0))
    assert(np.array_equal(res,[2.5000944373998295, 3.7773533038132356e-05, 0.0, 0.0, 0.0, 0.0]))

test_xyz_proj_mat()
test_lunation()
test_equatorial_to_ecliptic()
test_pbasis_to_elem()
