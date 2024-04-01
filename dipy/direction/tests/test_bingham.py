import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal)
from dipy.direction.bingham import (bingham_odf, bingham_fit_odf,
                                    _bingham_fit_peak, bingham_fiber_density,
                                    bingham_from_sh_new, _convert_bingham_pars,
                                    odi2k, k2odi)
from dipy.data import get_sphere

sphere = get_sphere('repulsion724')
sphere = sphere.subdivide(2)


def test_bingham_fit():
    """ Tests for bingham function and single Bingham fit"""
    peak_dir = np.array([1, 0, 0])
    ma_axis = np.array([0, 1, 0])
    mi_axis = np.array([0, 0, 1])
    k1 = 2
    k2 = 6
    f0 = 3

    # Test if maximum amplitude is in the expected Bingham main direction
    # which should be perpendicular to both ma_axis and mi_axis
    odf_test = bingham_odf(f0, k1, k2, ma_axis, mi_axis, peak_dir)
    assert_almost_equal(odf_test, f0)

    # Test Bingham fit on full sampled GT Bingham function
    odf_gt = bingham_odf(f0, k1, k2, ma_axis, mi_axis, sphere.vertices)
    a0, c1, c2, mu0, mu1, mu2 = _bingham_fit_peak(odf_gt, peak_dir, sphere, 45)

    # check scalar parameters
    assert_almost_equal(a0, f0, decimal=3)
    assert_almost_equal(c1, k1, decimal=3)
    assert_almost_equal(c2, k2, decimal=3)

    # check if measured peak direction and dispersion axis are aligned to their
    # respective GT
    Mus = np.array([mu0, mu1, mu2])
    Mus_ref = np.array([peak_dir, ma_axis, mi_axis])
    assert_array_almost_equal(np.abs(np.diag(np.dot(Mus, Mus_ref))),
                              np.ones(3))

    # check the same for bingham_fit_odf
    fits, n = bingham_fit_odf(odf_gt, sphere, max_search_angle=45)
    assert_almost_equal(fits[0][0], f0, decimal=3)
    assert_almost_equal(fits[0][1], k1, decimal=3)
    assert_almost_equal(fits[0][2], k2, decimal=3)
    Mus = np.array([fits[0][3], fits[0][4], fits[0][5]])
    # I had to decrease the precision in the assert below because main peak
    # direction is now calculated (before the GT direction was given)
    assert_array_almost_equal(np.abs(np.diag(np.dot(Mus, Mus_ref))),
                              np.ones(3), decimal=5)


def test_bingham_metrics():
    axis0 = np.array([1, 0, 0])
    axis1 = np.array([0, 1, 0])
    axis2 = np.array([0, 0, 1])
    k1 = 2
    k2 = 6
    f0_lobe1 = 3
    f0_lobe2 = 1

    # define the parameters of two bingham functions with different amplitudes
    fits = [(f0_lobe1, k1, k2, axis0, axis1, axis2)]
    fits.append((f0_lobe2, k1, k2, axis0, axis1, axis2))

    # TEST: Bingham Fiber density
    # As the amplitude of the first bingham function is 3 times higher than the
    # second, its integral have to be also 3 times larger.
    fd = bingham_fiber_density(fits)

    assert_almost_equal(fd[0]/fd[1], 3)

    # TEST: k2odi and odi2k conversions
    assert_almost_equal(odi2k(k2odi(k1)), k1)
    assert_almost_equal(odi2k(k2odi(k2)), k2)


def test_bingham_from_sh_new():

    # First test just to check right parameter conversion
    axis0 = np.array([1, 0, 0])
    axis1 = np.array([0, 1, 0])
    axis2 = np.array([0, 0, 1])
    k1 = 2
    k2 = 6
    f0_lobe1 = 3
    f0_lobe2 = 1

    fits = [(f0_lobe1, k1, k2, axis0, axis1, axis2)]
    fits.append((f0_lobe2, k1, k2, axis0, axis1, axis2))
    ref_pars = np.zeros((2, 12))
    ref_pars[0, 0] = f0_lobe1
    ref_pars[1, 0] = f0_lobe2
    ref_pars[0, 1] = ref_pars[1, 1] = k1
    ref_pars[0, 2] = ref_pars[1, 2] = k2
    ref_pars[0, 3:6] = ref_pars[1, 3:6] = axis0
    ref_pars[0, 6:9] = ref_pars[1, 6:9] = axis1
    ref_pars[0, 9:12] = ref_pars[1, 9:12] = axis2
    bpars = _convert_bingham_pars(fits)
    assert_array_almost_equal(bpars, ref_pars)

    # Reconstruct multi voxel ODFs to test bingham_from_sh_new
    ma_axis = np.array([0, 1, 0])
    mi_axis = np.array([0, 0, 1])
    k1 = 2
    k2 = 6
    f0 = 3
    odf = bingham_odf(f0, k1, k2, ma_axis, mi_axis, sphere.vertices)
    multi_odfs = np.zeros((2, 2, 1, len(sphere.vertices)))
    multi_odfs[...] = odf
    bpars = bingham_from_sh_new(multi_odfs, sphere)

    assert_almost_equal(bpars[0, 0, 0, 0, 0], f0, decimal=3)
    assert_almost_equal(bpars[0, 0, 0, 0, 1], k1, decimal=3)
    assert_almost_equal(bpars[0, 0, 0, 0, 2], k2, decimal=3)
    