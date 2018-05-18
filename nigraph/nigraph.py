#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Commonly used utility functions."""

# mainly backports from future numpy here

from __future__ import absolute_import, division, print_function
import numpy as np
import nibabel as nib


def thresholding_abs(A, thr, smaller=True, copy=True):
    """thresholding of the adjacency matrix with an absolute value.

    Parameters
    ----------
    A : ndarray, shape(n, n) **or** shape(n_tps, n, n)
        Adjacency matrix of the graph. The matrix must be weighted with zero
        values along the main diagonal.
    thr : float
        Absolute threshold. Edges with weights `</>` to the given threshold
        value will be removed.
    smaller : boolean (default=True)
        Threshold values smaller than the given threshold. If ``smaller=False``
        values greater
        than are thresholded.
    copy : boolean
        Whether to copy the input or change the matrix in-place (default=True).

    Returns
    -------
    A_thr : ndarray, shape(n, n) **or** shape(n_tps, n, n)
        Thresholded adjacency matrix.

    Notes
    -----
    Supports also thresholding of **dynamic graphs**.

    See also
    --------
    thresholding_rel, thresholding_pval, thresholding_M, thresholding_max

    Examples
    --------
    >>> d = get_fmri_data()
    >>> # p-values are not required for thresholding with an given value
    >>> A = adj_static(d, pval=False)
    >>> A_thr = thresholding_abs(A, .3)
    >>> # smallest nonzero edge weight after thresholding
    >>> print A_thr[A_thr > 0].min()
    0.30012873644
    """

    # this function was tested against BCT and gives the same
    # results as the matlab function: 'threshold_absolute.m'

    # TODO np.clip is faster than inplace operation:
    # A = np.random.normal(size=10000*10000).reshape((10000,10000))
    # In [30]: %timeit np.clip(A,0,np.inf, A)
    # 1 loops, best of 3: 315 ms per loop
    #
    # In [31]: %timeit A[A < 0]=0
    # 1 loops, best of 3: 638 ms per loop

    def _thr(data, smaller=True):
        if smaller:
            data[data < thr] = 0.
        else:
            data[data > thr] = 0.
        return data

    if copy:
        data = A.copy()
        return _thr(data)
    else:
        return _thr(A)


def load_mri(func, mask):
    """load MRI voxel data

    The data is converted into a 2D (n_voxel, n_tps) array.

    Parameters
    ----------
    func : string
        Path to imaging data (e.g. nifti).
    mask : string
        Path to binary mask (e.g. nifti) that defines brain regions. Values > 0
        are regarded as brain tissue.

    Returns
    -------
    ts : ndarray, shape(n_voxel, n_tps)
        Timeseries information in a 2D array.

    See Also
    --------
    save_mri: save MRI voxel data to disk.

    Examples
    --------
    >>> ts = load_mri(func='localizer.nii.gz', mask='V1_mask.nii.gz')
    """

    # load mask data
    m = nib.load(mask).get_data()

    # load func data
    d = nib.load(func).get_data()

    # mask the data
    func_data = d[m != 0]
    # nib.load(func).get_data()[nib.load(mask).get_data()!=0]

    del d

    return func_data


def save_mri(data, mask, fname=None):
    """save MRI voxel data

    Parameters
    ----------
    data : ndarray, shape(n_voxel,) **or** shape(n_voxel, n_tps)
       Voxel data to save to disk.
    mask : string
        Path to binary mask (e.g. nifti) that defines brain regions. Values > 0
        are regarded as brain tissue.
    fname : string
        Filename.

    Examples
    --------
    >>> ts = load_mri(func='localizer.nii.gz', mask='V1_mask.nii.gz')
    >>> ts = ts + 1. # some operation
    >>> save_mri(ts, 'V1_mask.nii.gz', 'localizer_plus_one.nii.gz')
    """
    # load mask data
    f = nib.load(mask)
    m = f.get_data()
    aff = f.get_affine()

    s = m.shape
    if len(data.shape) == 2:
        n_tps = data.shape[1]
    else:
        n_tps = 1
        data = data[:, np.newaxis]

    res = np.zeros((s[0], s[1], s[2], n_tps))  # + time
    res[m != 0] = data

    # save to disk
    if fname is not None:
        nib.save(nib.Nifti1Image(res, aff), fname)


def load_roi_mri(func, mask):
    """returns mean-timeseries based on a provided ROI-mask

    Parameters
    ----------
    func : string
        imaging-file (e.g. nifti). Fuctional imaging data that contains the 3D+
        time information (n_tps)
    mask : string
        imaging-file (e.g. nifti). ROIs are defined as areas with the same mask
        value; all values < 1 are discarded

    Returns
    -------
    ts : ndarray, shape(n_rois, n_tps)
        Timeseries information in a 2D array.

    Notes
    -----
    Mask values don't need to have ascending or descending order, but the
    returned array is always sorted in ascending order.

    See Also
    --------
    save_roi_mri: save ROI data

    References
    ----------
    .. [1] Craddock, R. C., James, G. A., Holtzheimer, P. E., III, Hu, X. P., &
           Mayberg, H. S. (2011). A whole brain fMRI atlas generated via
           spatially constrained spectral clustering. Human Brain Mapping,
           n/a–n/a. doi:10.1002/hbm.21333
    .. [2] Goulas, A., Uylings, H. B. M., & Stiers, P. (2012). Unravelling the
           intrinsic functional organization of the human lateral frontal
           cortex: a parcellation scheme based on resting state fMRI. Journal
           of Neuroscience, 32(30), 10238–10252.
           doi:10.1523/JNEUROSCI.5852-11.2012

    Examples
    --------
    >>> # TODO
    >>> ts = load_roi_mri(func, mask, mode='mean')
    """

    # load mask data
    m = nib.load(mask).get_data()

    # load func data
    d = nib.load(func).get_data()
    n_samples = d.shape[-1]

    if len(m.shape) > 3:
        m = m.reshape(m.shape[0], m.shape[1], m.shape[2])

    if not d.shape[:3] == m.shape:
        raise ValueError('The functional data and the given mask are not in \
                         the same reference space')

    # mask_data = m[m != 0]
    # uni_rois = np.unique(mask_data) # without zero
    # n_rois = uni_rois.size

    uni_rois = np.unique(m)[1:]  # without zero
    n_rois = uni_rois.size

    ts_data = np.empty((n_rois, n_samples))
    roi_counter = 0

    # this also works with mask_indices that are not ascending;
    # range(n_rois) does not

    # TODO: faster when transform 4D --> 2D array!
    # BUT only for n_rois > 400; for n_rois = 4000 -> 4 x faster
    # but the transformation generates a huge overhead

    # mm = m!=0
    # mask_2d = m[mm]
    # data_2d = d[mm]
    #
    # if mode is 'mean':
    #     for i in uni_rois:
    #         ts_data[roi_counter,:] = np.mean(data_2d[mask_2d == i,:], axis=0)
    #         roi_counter += 1

    for i in uni_rois:
        ts_data[roi_counter, :] = np.mean(d[m == i, :], axis=0)
        roi_counter += 1

    del d

    return ts_data


def save_roi_mri(data, mask, fname='roi_data.nii.gz', sort=None):
    """saves ROI data (e.g. local graph metrics) to imaging file

    Parameters
    ----------
    data : ndarray, shape(n,) **or** shape(n, n_tps)
       Local graph metric that corresponds to the ROIs defined in the mask file
    mask : string
        Imaging-file (e.g. nifti). ROIs are defined as areas with the same
        unique mask values. Only mask values > 1 are regarded as brain tissue.
    fname : string
        Filename (default='roi_data.nii.gz').
    sort : ndarray, shape(n,), optional
        Integer providing the mapping between data and mask. If no mapping is
        provided, it is assumed to be in ascending order of unique mask values.
        #TODO carefully check sort parameter

    Notes
    -----
    If ``sort=None`` the mapping between data values and mask is assumed to be
    ``data[0]=np.unique(mask[mask!=0])[0]``

    See Also
    --------
    load_roi_mri: load ROI data

    Examples
    --------
    >>> _, func_path = get_fmri_rss_data()
    >>> _, mask_path, labels, coords = aal(n=116, space='3mm')
    >>> data = load_roi_mri(func_path, mask_path)
    >>> A = adj_static(data)
    >>> A[A<0.1] = 0
    >>> k = degree(A)
    >>> print k.shape
    (116,)
    >>> # save local degree k to nifti file
    >>> save_roi_mri(k, mask_path, fname='local_degree.nii.gz')
    """

    # load mask data
    f = nib.load(mask)
    m = f.get_data()
    aff = f.get_affine()
    uni_rois = np.unique(m[m != 0])  # without zeros

    if data.ndim == 2:
        n_tps = data.shape[1]
    else:
        n_tps = 1
        data = data[:, np.newaxis]

    if not uni_rois.size == data.shape[0]:
        raise ValueError('The number of nodes provided by data and mask do not\
                         match')

    xdim, ydim, zdim = m.shape
    res = np.zeros((xdim, ydim, zdim, n_tps))  # + time
    roi_counter = 0
    for i in uni_rois:
        res[m == i] = data[roi_counter, :]
        roi_counter += 1

    if sort is not None:
        res = res[sort, :]

    # save to disk
    if fname is not None:
        nib.save(nib.Nifti1Image(res, aff), fname)
