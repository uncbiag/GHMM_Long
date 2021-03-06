"""
Step 2:
Estimating the hidden variables, b_i(v), v = 1,2,..., num_vox, based on HMRF.
"""

import numpy as np
from dipy.sims.voxel import add_noise
from dipy.segment.mrf import ConstantObservationModel, IteratedConditionalModes


class RegionDetectionHMRF(object):
    r"""
    This class contains the methods for region detection using the Markov
    Random Fields modeling approach
    """

    def __init__(self, save_history=False, verbose=True):

        self.save_history = save_history
        self.segmentations = []
        self.pves = []
        self.energies = []
        self.energies_sum = []
        self.verbose = verbose

    def classify(self, image, nclasses, gamma):
        r"""
        This method uses the Maximum a posteriori - Markov Random Field
        approach for segmentation by using the Iterative Conditional Modes and
        Expectation Maximization to estimate the parameters.

        Parameters
        ----------
        image : ndarray,
                2D or 3D structural image.
        nclasses : int,
                number of desired classes.
        gamma : float,
                smoothing parameter, the higher this number the smoother the
                output will be.

        Returns
        -------
        initial_segmentation : ndarray,
                segmented image with all tissue types
                specified in nclasses.
        final_segmentation : ndarray,
                final refined segmentation containing all
                tissue types.
        PVE : ndarray,
                probability map of each tissue type.
        """

        nclasses = nclasses + 1  # One extra class for the background
        energy_sum = [1e-05]

        com = ConstantObservationModel()
        icm = IteratedConditionalModes()

        if image.max() > 1:
            image = np.interp(image, [0, image.max()], [0.0, 1.0])

        mu, sigma = com.initialize_param_uniform(image, nclasses)
        p = np.argsort(mu)
        mu = mu[p]
        sigma = sigma[p]
        sigmasq = sigma ** 2

        neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
        seg_init = icm.initialize_maximum_likelihood(neglogl)

        mu, sigma = com.seg_stats(image, seg_init, nclasses)
        sigmasq = sigma ** 2

        zero = np.zeros_like(image) + 0.001
        zero_noise = add_noise(zero, 10000, 1, noise_type='gaussian')
        image_gauss = np.where(image == 0, zero_noise, image)

        final_segmentation = np.empty_like(image)
        initial_segmentation = seg_init.copy()

        max_iter = 100
        tolerance = 1e-05

        for i in range(max_iter):

            if self.verbose:
                print('>> Iteration: ' + str(i))

            PLN = icm.prob_neighborhood(seg_init, beta, nclasses)
            PVE = com.prob_image(image_gauss, nclasses, mu, sigmasq, PLN)

            mu_upd, sigmasq_upd = com.update_param(image_gauss,
                                                       PVE, mu, nclasses)
            ind = np.argsort(mu_upd)
            mu_upd = mu_upd[ind]
            sigmasq_upd = sigmasq_upd[ind]

            negll = com.negloglikelihood(image_gauss, mu_upd, sigmasq_upd, nclasses)
            final_segmentation, energy = icm.icm_ising(negll, beta, seg_init)
            energy_sum.append(energy[energy > -np.inf].sum())

            if self.save_history:
                self.segmentations.append(final_segmentation)
                self.pves.append(PVE)
                self.energies.append(energy)
                self.energies_sum.append(energy[energy > -np.inf].sum())

            if i % 10 == 0 and i != 0:

                tol = tolerance * (np.amax(energy_sum) - np.amin(energy_sum))

                test_dist = np.absolute(np.amax(energy_sum[np.size(energy_sum) - 5: i]) -
                                        np.amin(energy_sum[np.size(energy_sum) - 5: i]))

                if test_dist < tol:

                    break

            seg_init = final_segmentation.copy()
            mu = mu_upd.copy()
            sigmasq = sigmasq_upd.copy()

        PVE = PVE[..., 1:]

        return initial_segmentation, final_segmentation, PVE
