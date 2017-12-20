"""
Step 2:
Estimating the hidden variables, b_i(v), v = 1,2,..., num_vox, based on MRF-MAP.
"""

import numpy as np


def ind2sub(array_shape, ind, size):
    ind[ind < 0] = -1
    if size == 2:
        ind[ind >= array_shape[0]*array_shape[1]] = -1
        rows = (ind.astype('int') / array_shape[1])
        cols = ind % array_shape[1]
        sub = [rows, cols]
    else:
        ind[ind >= array_shape[0] * array_shape[1] * array_shape[2]] = -1
        rows = (ind.astype('int') / (array_shape[1] * array_shape[2]))
        ind_r = ind % (array_shape[1] * array_shape[2])
        cols = (ind_r.astype('int') / array_shape[2])
        depths = ind_r % array_shape[2]
        sub = [rows, cols, depths]
    return sub


def sub2ind(array_shape, sub):
    if len(array_shape) == 2:
        ind = sub[0] * array_shape[1] + sub[1]
        ind[ind < 0] = -1
        ind[ind >= array_shape[0] * array_shape[1]] = -1
    else:
        ind = sub[0]*array_shape[1]*array_shape[2] + sub[1]*array_shape[1] + sub[2]
        ind[ind < 0] = -1
        ind[ind >= array_shape[0] * array_shape[1] * array_shape[2]] = -1
    return ind


def delta_fun(x1, x2):
    if x1 == x2:
        z = 1
    else:
        z = 0
    return z


def map_fun(b_0, x_design, y_design, dx, template, beta, mu, s2, gamma, nclasses, map_iter):
    y_design1 = y_design[dx > 0, :]
    x_design1 = x_design[dx > 0, :]
    res_mat = y_design1 - dot(x_design1, beta)
    num_sub = res_mat.shape[0]
    num_vox = res_mat.shape[1]

    for ii in range(num_sub):
        r_i = res_mat[ii, :]
        mu_i = mu[ii, :]
        s2_i = s2[ii]
        b0_i = b_0[ii, :]
        u = 0
        for jj in range(map_iter):
            u1 = zeros(num_vox, nclasses)
            u2 = zeros(num_vox, nclasses)
            for ll in range(nclasses):
                temp_i = (r_i - mu_i[ll])**2/s2_i/2+np.log(s2_i)/2
                u1[:, ll] = u1[:, l]+temp_i

                for vii in range(num_vox):
                    u3 = np.zeros(num_vox)
                    if len(template.shape) == 2:
                        vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
                        ind_list = np.where(vec_template == 1)
                        ind = ind_list[vii]
                        sub2 = ind2sub(template.shape, ind, 2)
                        ind2_1 = sub2ind(template.shape, [sub2[0] + 1, sub[1]])
                        ind2_2 = sub2ind(template.shape, [sub2[0], sub2[1] + 1])
                        ind2_3 = sub2ind(template.shape, [sub2[0] - 1, sub2[1]])
                        ind2_4 = sub2ind(template.shape, [sub2[0], sub2[1] - 1])
                        u3[vii] = delta_fun(ll, b0_i[ind2_1]) + delta_fun(ll, b0_i[ind2_2])\
                             + delta_fun(ll, b0_i[ind2_3]) + delta_fun(ll, b0_i[ind2_4])
                    else:
                        vec_template = np.reshape(template, (template.shape[0] * template.shape[1]
                                                             * template.shape[2], 1))
                        ind_list = np.where(vec_template == 1)
                        ind = ind_list[vii]
                        sub3 = ind2sub(template.shape, ind, 3)
                        ind3_1 = sub2ind(template.shape, [sub3[0] + 1, sub3[1], sub3[2]])
                        ind3_2 = sub2ind(template.shape, [sub3[0], sub3[1] + 1, sub3[2]])
                        ind3_3 = sub2ind(template.shape, [sub3[0], sub3[1], sub3[2] + 1])
                        ind3_4 = sub2ind(template.shape, [sub3[0] - 1, sub3[1], sub3[2]])
                        ind3_5 = sub2ind(template.shape, [sub3[0], sub3[1] - 1, sub3[2]])
                        ind3_6 = sub2ind(template.shape, [sub3[0], sub3[1], sub3[2] - 1])
                        u3[vii] = delta_fun(ll, b0_i[ind3_1]) + delta_fun(ll, b0_i[ind3_2]) + \
                             delta_fun(ll, b0_i[ind3_3]) + delta_fun(ll, b0_i[ind3_4]) \
                             + delta_fun(ll, b0_i[ind3_5]) + delta_fun(ll, b0_i[ind3_6])
                    u2[vii, ll] = np.sum(u3)
            u = u1 + u2*gamma
            b0_i = np.argmin(u, axis=1)-1

        if len(template.shape) == 2:
            vec_template = np.reshape(template, (template.shape[0] * template.shape[1], 1))
        else:
            vec_template = np.reshape(template, (template.shape[0] * template.shape[1]
                                                 * template.shape[2], 1))
        b_0[i, vec_template == 1] = b0_i
        u_i = np.exp(-np.min(u, axis=1))
        u_i0 = u_i[b0_i == 0]
        r_i0 = r_i[b0_i == 0]
        mu[ii, 0] = np.sum(u_i0*r_i0)/np.sum(u_i0)
        u_i1 = u_i[b0_i == 1]
        r_i1 = r_i[b0_i == 1]
        mu[ii, 1] = np.sum(u_i1 * r_i1) / np.sum(u_i1)

    """return b_0, mu"""
    return b_0, mu
