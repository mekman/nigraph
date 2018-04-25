from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy as np
import numpy.testing as npt
import nigraph as ng


def test_thresholding_abs():
    A = np.identity(3)
    A_thr = ng.thresholding_abs(A, 1)
    npt.assert_equal(A, A_thr)
