from __future__ import absolute_import, division, print_function
# import os.path as op
import numpy as np
import numpy.testing as npt
from nigraph import utilities


def test_is_directed():
    A = np.random.normal(size=(4, 4))
    res = utilities.is_directed(A)
    npt.assert_equal(res, False)
