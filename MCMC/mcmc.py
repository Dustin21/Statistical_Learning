import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd



class DataDistribution(object):
    def __init__(self, means, sigmas, weights):
        """
        Initialize class variables
        :param means: vector of means of dimension d for each distribution in mixture
        :param sigmas: vector of variances of dimension d for each distribution in mixture
        :param weights: vector of weights of dimension d for each distribution in mixture
        """
        self.means = means
        self.sigmas = sigmas
        self.weights = weights

    def gaussian_mixture(self, x):
        """
        Generates a Gaussian mixture distribution
        :param x: the quantile of the gaussian with which to sample the pdf
        :return: x_out returns the probability density from the quantile x
        """
        def normpdf(x, mu, sigma):
            """
            The pdf of the normal distribution
            :param x: quantile on which to sample the density function
            :param mu: scalar vector of mean of each of the Gaussians
            :param sigma: scalar vector of variance of each of the Gaussians
            :return: the value of the pdf for a simple Gaussian
            """
            u = (x - mu) / abs(sigma)
            y = (1 / (np.sqrt(2 * np.pi) * abs(sigma))) * np.exp(-u * u / 2)
            return y
        l = len(self.means)
        x_out = 0

        # build the mixture via a weights combination of normpdf
        for i in range(l):
            x_out = x_out + self.weights[i] * normpdf(x, self.means[i], self.sigmas[i])
        return x_out


class MCMC(object):
    def __init__(self, x0, sigma, n):
        """
        Initialize class variables
        :param x0: initialized scalar point
        :param sigma: the variance of the MCMC proposal distribution
        :param n: the number of samples of MCMC
        """
        self.x0 = x0
        self.sigma = sigma
        self.n = n

    def mcmc(self, p: object):
        """
        Performs MCMC sampling on the Gaussian mixture
        :param p: the gaussian_mixture function
        :return: the samples from the distribution
        """
        t = time.time() # initialize start time
        x_old = self.x0
        x = []

        # the magic happens here - MCMC algo
        for i in range(self.n):

            x_new = np.random.normal(x_old, self.sigma)
            p_new = p(x_new)
            p_old = p(x_old)
            U = np.random.uniform()

            A = p_new / p_old

            if U < A:
                x_old = x_new

            x.append(x_old)

        burn_in = int(len(x)*0.05)
        del x[-0:burn_in]

        print("Elapsed time: %1.2f seconds" % (time.time() - t))

        return x

    def plotHistorgram(self, p, x, xrange, label=u'MCMC distribution'):
        """
        Plots the true density function with the empirical density from
        the MCMC samples
        :param p: the gaussian_mixture function
        :param x: the samples output from mcmc
        :param xrange: the range of the x-axis of the plot
        :param label: the title of the plot
        :return:
        """
        # plot sample histogram
        plt.hist(x, 100, alpha=0.7, label=u'MCMC distribution', normed=True)

        # plot the true function
        xx = np.linspace(xrange[0], xrange[1], 100)
        plt.plot(xx, p(xx), 'r', label=u'True distribution')
        plt.legend()
        plt.show()

        print("Starting point was ", x[0])


    def plotFirstMomentConvergence(self, x):
        """
        Plots the convergence of the first moment E(X) over
        MCMC iterations.
        :param x: MCMC samples from the distribution
        :return: E(X) convergence plot
        """
        x_in = np.transpose(x)
        cummean = np.cumsum(x_in) / np.arange(1, len(x_in) + 1)
        plt.plot(cummean, label=u'E(X) Convergence', color='k', linewidth=1.5)
        return cummean

    def plotSecondMomentConvergence(self, x):
        """
        Plots the convergence of the second moment E(X^2), or
        more precisely, the standard deviation over MCMC iterations.
        :param x: MCMC samples from distribution
        :return: E(X^2) convergence plot
        """
        x_in = np.transpose(x)
        cumstd = pd.expanding_std(x_in, min_periods=1)
        plt.plot(cumstd, label = u'E(X^2) Convergence', color='k', linewidth=1.5)
        return cumstd