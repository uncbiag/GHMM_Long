"""
Step 1.1:
Estimating beta and sigma, v = 1,2,..., num_vox, based on normal subjects.
"""

import numpy as np
from numpy import dot
from numpy.linalg import inv, norm


def lmm_fun(y_design, x_design, dx):
    y_design0 = y_design[dx == 0, :]
    x_design0 = x_design[dx == 0, :]
    p = x_design0.shape[1]
    beta = np.zeros(p, num_vox)
    s2 = np.zeros(p)
    for v in range(num_vox):
        beta[:, v] = dot(inv(dot(x_design0.T, x_design0)), dot(x_design0.T, np.atleast_2d(y_design0[:, v]).T))
        ort_mat = np.eye(x_design0.shape[0])-dot(dot(x_design0, inv(dot(x_design0.T, x_design0))), x_design0.T)
        s2[v] = norm(dot(ort_mat, np.atleast_2d(y_design0[:, v]).T))**2/(x_design0.shape[0]-p)

    """return mu, alpha"""
    return beta, sigma
