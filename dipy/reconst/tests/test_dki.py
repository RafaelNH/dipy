""" Testing DKI """

from __future__ import division, print_function, absolute_import

import numpy as np

from numpy.testing import assert_array_almost_equal

from dipy.sims.voxel import multi_tensor_dki

import dipy.reconst.dki as dki

from dipy.reconst.dki import (mean_kurtosis, carlson_rf,  carlson_rd)

from dipy.io.gradients import read_bvals_bvecs

from dipy.core.gradients import gradient_table

from dipy.data import get_data

from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)

from dipy.core.sphere import Sphere


fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

# 2 shells for techniques that requires multishell data
bvals_2s = np.concatenate((bvals, bvals * 2), axis=0)
bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
gtab_2s = gradient_table(bvals_2s, bvecs_2s)

# Simulation 1. signals of two crossing fibers are simulated
mevals_cross = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                         [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
angles_cross = [(80, 10), (80, 10), (20, 30), (20, 30)]
fie = 0.49
frac_cross = [fie*50, (1-fie) * 50, fie*50, (1-fie) * 50]

# Noise free simulates
signal_cross, dt_cross, kt_cross = multi_tensor_dki(gtab_2s, mevals_cross,
                                                    S0=100,
                                                    angles=angles_cross,
                                                    fractions=frac_cross,
                                                    snr=None)

evals_cross, evecs = decompose_tensor(from_lower_triangular(dt_cross))
crossing_ref = np.concatenate((evals_cross, evecs[0], evecs[1], evecs[2],
                               kt_cross), axis=0)

# Simulation 2. Spherical kurtosis tensor.- for white matter, this can be a
# biological implaussible scenario, however this simulation is usefull for
# testing the estimation of directional apparent kurtosis and the mean
# kurtosis, since its directional and mean kurtosis ground truth are a constant
# which can be easly mathematicaly calculated

# simulate a spherical kurtosis tensor
Di = 0.00099
De = 0.00226
mevals_sph = np.array([[Di, Di, Di], [De, De, De]])
frac_sph = [50, 50]
signal_sph, dt_sph, kt_sph = multi_tensor_dki(gtab_2s, mevals_sph, S0=100,
                                              fractions=frac_sph,
                                              snr=None)
evals_sph, evecs = decompose_tensor(from_lower_triangular(dt_sph))
params_sph = np.concatenate((evals_sph, evecs[0], evecs[1], evecs[2], kt_sph),
                            axis=0)

# Compute ground truth. Since KT is spherical, appparent kurtosic coeficient
# for all gradient directions and mean kurtosis have to be equal to Kref_sph.
f = 0.5
Dg = f*Di + (1-f)*De
Kref_sphere = 3 * f * (1-f) * ((Di-De) / Dg) ** 2

# Simulation 3. Multi-voxel simulations - dataset of four voxels is simulated.
# Since the objective of this simulation is to see if procedures are able to
# work with multi-dimentional data all voxels contains the same crossing signal
# produced in simulation 1.

DWI = np.zeros((2, 2, 1, len(gtab_2s.bvals)))
DWI[0, 0, 0] = DWI[0, 1, 0] = DWI[1, 0, 0] = DWI[1, 1, 0] = signal_cross

multi_params = np.zeros((2, 2, 1, 27))
multi_params[0, 0, 0] = multi_params[0, 1, 0] = crossing_ref
multi_params[1, 0, 0] = multi_params[1, 1, 0] = crossing_ref


def test_dki_fits():
    """ DKI fits are tested on noise free crossing fiber simulates """

    # OLS fitting
    dkiM = dki.DiffusionKurtosisModel(gtab_2s, fit_method="OLS")
    dkiF = dkiM.fit(signal_cross)

    assert_array_almost_equal(dkiF.model_params, crossing_ref)

    # WLS fitting
    dki_wlsM = dki.DiffusionKurtosisModel(gtab_2s, fit_method="WLS")
    dki_wlsF = dki_wlsM.fit(signal_cross)

    assert_array_almost_equal(dki_wlsF.model_params, crossing_ref)

    # testing multi-voxels
    dkiF_multi = dkiM.fit(DWI)
    assert_array_almost_equal(dkiF_multi.model_params, multi_params)

    dkiF_multi = dki_wlsM.fit(DWI)
    assert_array_almost_equal(dkiF_multi.model_params, multi_params)


def test_apparent_kurtosis_coef():
    """ Apparent kurtosis coeficients are tested for a spherical kurtosis
    tensor """

    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    AKC = dki.apparent_kurtosis_coef(params_sph, sph)

    # check all direction
    for d in range(len(gtab.bvecs[gtab.bvals > 0])):
        assert_array_almost_equal(AKC[d], Kref_sphere)


def test_dki_predict():
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)
    pred = dkiM.predict(crossing_ref, S0=100)

    assert_array_almost_equal(pred, signal_cross)

    # just to check that it works with more than one voxel:
    pred_multi = dkiM.predict(multi_params, S0=100)
    assert_array_almost_equal(pred_multi, DWI)


def test_mk(): 
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)
    dkiF = dkiM.fit(signal_cross)

    # MK analytical solution
    MK_as = dkiF.mk

    # MK numerical method
    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    MK_nm = mean_kurtosis(dkiF.model_params, sph)

    assert_array_almost_equal(MK_as, MK_nm, decimal=1)


def test_carlson_rf():
    
    # Define inputs that we know the outputs from:
    # Carlson, B.C., 1994. Numerical computation of real or complex
    # elliptic integrals. arXiv:math/9409227 [math.CA]
    
    # Real values
    x = np.array([1.0, 0.5, 2.0])
    y = np.array([2.0, 1.0, 3.0])
    z = np.array([0.0, 0.0, 4.0])
    
    # Defene reference outputs
    RF_ref = np.array([1.3110287771461, 1.8540746773014, 0.58408284167715])
    
    # Compute integrals
    RF =  carlson_rf(x, y, z)

    # Compare
    assert_array_almost_equal(RF, RF_ref)
    
    # Complex values
    x = np.array([1j, 1j - 1, 1j, 1j - 1])
    y = np.array([-1j, 1j, -1j, 1j])
    z = np.array([0.0, 0.0, 2, 1 - 1j])
    
    # Defene reference outputs
    RF_ref = np.array([1.8540746773014, 0.79612586584234 - 1.2138566698365j,
                       1.0441445654064, 0.93912050218619 - 0.53296252018635j])
    # Compute integrals
    RF =  carlson_rf(x, y, z, errtol=3e-5)

    # Compare
    assert_array_almost_equal(RF, RF_ref)


def test_carlson_rd():
    
    # Define inputs that we know the outputs from:
    # Carlson, B.C., 1994. Numerical computation of real or complex
    # elliptic integrals. arXiv:math/9409227 [math.CA]
    
    # Real values
    x = np.array([0.0, 2.0])
    y = np.array([2.0, 3.0])
    z = np.array([1.0, 4.0])
    
    # Defene reference outputs
    RD_ref = np.array([1.7972103521034, 0.16510527294261])
    
    # Compute integrals
    RD =  carlson_rd(x, y, z, errtol=1e-5)

    # Compare
    assert_array_almost_equal(RD, RD_ref)
    
    # Complex values
    x = np.array([1j, 0.0, 0.0, -2 - 1j])
    y = np.array([-1j, 1j, 1j-1, -1j])
    z = np.array([2.0, -1j, 1j, -1 + 1j])
    
    # Defene reference outputs
    RD_ref = np.array([0.65933854154220, 1.2708196271910 + 2.7811120159521j,
                       -1.8577235439239 - 0.96193450888839j, 
                       1.8249027393704 - 1.2218475784827j])
    # Compute integrals
    RD =  carlson_rd(x, y, z, errtol=1e-5)

    # Compare
    assert_array_almost_equal(RD, RD_ref)


def wls_fit_dki(design_matrix, data, min_signal=1):
    r"""
    Adaption of the WLS fit implemented by Maurizio with faster all voxel
    lopping, with new output format (all important KT elements saved).
    """

    tol = 1e-6
    if min_signal <= 0:
        raise ValueError('min_signal must be > 0')

    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    # dki_params = np.empty((len(data_flat), 6, 3))
    # new line:
    dki_params = np.empty((len(data_flat), 27))
    min_diffusivity = tol / -design_matrix.min()

    ols_fit = _ols_fit_matrix(design_matrix)

    # for param, sig in zip(dki_params, data_flat):
    #     param[0], param[1:4], param[4], param[5] = _wls_iter(ols_fit,
    #     design_matrix, sig, min_signal, min_diffusivity)
    # new line:
    for vox in range(len(data_flat)):
        dki_params[vox] = _wls_iter(ols_fit, design_matrix, data_flat[vox],
                                    min_signal, min_diffusivity)

    # dki_params.shape=data.shape[:-1]+(18,)
    # dki_params=dki_params
    return dki_params


def _ols_fit_matrix(design_matrix):
    """
    (implemented by Maurizio)
    Helper function to calculate the ordinary least squares (OLS)
    fit as a matrix multiplication. Mainly used to calculate WLS weights. Can
    be used to calculate regression coefficients in OLS but not recommended.

    See Also:
    ---------
    wls_fit_tensor, ols_fit_tensor

    Example:
    --------
    ols_fit = _ols_fit_matrix(design_mat)
    ols_data = np.dot(ols_fit, data)
    """

    U, S, V = np.linalg.svd(design_matrix, False)
    return np.dot(U, U.T)


def _wls_iter(ols_fit, design_matrix, sig, min_signal, min_diffusivity):
    ''' Helper function used by wls_fit_tensor.
    '''
    sig = np.maximum(sig, min_signal)  # throw out zero signals
    log_s = np.log(sig)
    w = np.exp(np.dot(ols_fit, log_s))
    result = np.dot(np.linalg.pinv(design_matrix * w[:, None]), w * log_s)
    D = result[:6]
    # tensor=from_lower_triangular(D)
    # new line
    evals, evecs = decompose_tensor(from_lower_triangular(D),
                                    min_diffusivity=min_diffusivity)

    # MeanD_square=((tensor[0,0]+tensor[1,1]+tensor[2,2])/3.)**2
    # new_line:
    MeanD_square = (evals.mean(0))**2
    K_tensor_elements = result[6:21] / MeanD_square

    # new line:
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 K_tensor_elements), axis=0)

    out_shape = sig.shape[:-1] + (-1, )
    dki_params = dki_params.reshape(out_shape)

    # return decompose_tensors(tensor, K_tensor_elements,
    #                          min_diffusivity=min_diffusivity)
    # line line:
    return dki_params
