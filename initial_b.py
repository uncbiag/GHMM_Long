"""
Step 1.2:
Initial the diseased regions from patients (k-means)
"""

import numpy as np
from numpy import dot
from sklearn.cluster import KMeans


def initial_b_fun(y_design, x_design, dx, template, beta, nclasses):
    y_design1 = y_design[dx > 0, :]
    x_design1 = x_design[dx > 0, :]
    res_mat = y_design1-dot(x_design1, beta)
    if len(template.shape) == 2:
        b_0 = 2*np.ones(shape=(res_mat.shape[0], template.shape[0]*template.shape[1]))
        vec_template = np.reshape(template, (template.shape[0]*template.shape[1], 1))
    else:
        b_0 = 2*np.ones(shape=(res_mat.shape[0], template.shape[0]*template.shape[1]*template.shape[2]))
        vec_template = np.reshape(template, (template.shape[0]*template.shape[1]*template.shape[2], 1))
    mu = np.zeros(shape=(res_mat.shape[0], nclasses))
    for i in range(x_design1.shape[0]):
        kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(res_mat[i, :])
        centers = kmeans.cluster_centers
        if centers[0] > centers[1]:
            b_0[i, vec_template == 1] = 1-kmeans.labels_
            mu[i, 0] = centers[1]
            mu[i, 1] = centers[0]
        else:
            b_0[i, vec_template == 1] = kmeans.labels_
            mu[i, :] = centers

    """return b_0, mu"""
    return b_0, mu
