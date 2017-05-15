import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal as mvnorm


class GaussianMixture(object):
	def __init__(self, pi, centroids, covmat):
		"""
		Initialize gaussianMixture parameters
		:param pi: mixture coefs
		:param centroids: mean vector
		:param covmat: covariance matrix array
		"""
		self.pi = pi
		self.centroids = centroids
		self.covmat = covmat

	def simulate_gm(self, n_samples):
		"""
		Simulate from a Gaussian mixture
		:type n_samples: number of samples to draw
		:return: simulation output
		"""
		# initialise counting params
		K = len(self.pi)
		D = self.centroids.shape[1]
		N = n_samples
		pi_cum = np.cumsum(self.pi)

		# initialize empty matrix
		x_out = np.zeros([N, D])

		# draw each sample from the mixture according
		# to the first occurring instance where a uniform
		# random draw is less that a given cumulative weight
		for n in range(N):
			u = np.random.uniform() # draw uniform
			q = u < pi_cum # is uniform draw < pi?
			for j in range(K):
				if q[j] == True:
					x_out[n, :] = np.random.multivariate_normal(self.centroids[j], np.diag(self.covmat[j]))
					break
				else:
					None
		return x_out

	def pdf_gm(self, X):
		"""
		Produce the probability density function of the GM
		:param X: matrix of data, n obs * k variables
		:return: density of gaussian mixture
		"""
		# initialize count K
		K = len(self.pi)

		# iterate over a weighted sum of normal pdf's
		mix = 0
		for k in range(K):
			mix = mix + self.pi[k] * mvnorm.pdf(X, self.centroids[k], np.diag(self.covmat[k]))
		return mix

	def surface_gm(self, ranges, fineness, filename, labels=None):
		"""
		Surface plot of gaussian mixture
		:param mix_out: output of method pdf_gm
		:param range: 2 by 1 vector of plot range
		:param grid: fineness of plot mesh
		:param labels: variable labels
		:return: na
		"""
		if self.centroids.shape[1] > 2:
			print("Only bivariate data allowed for surface_gm.")

		def mixture_pdf(X, Y, pi, centroids, covmat):
			"""
			Bivariate pdf of gaussian mixture
			:param X: data vector for variable x
			:param Y: data vector for variable y
			:param pi: mixture parameter
			:param centroids: means for multivariate normal
			:param covmat: covariance matrix for multivariate normal
			:return: pdf of MVN
			"""
			# initialize counting params
			K = len(pi)
			D = centroids.shape[1]
			N = len(X)

			# sum over weighted bivariate normal pdf's
			mix = np.zeros([N, N])
			for k in range(K):
				mix = mix + pi[k] * mlab.bivariate_normal(X, Y, covmat[k, 0], covmat[k, 1], centroids[k, 0], centroids[k, 1], 0)
			return mix

		# create grid for 3D surface plot
		X = np.arange(ranges[0], ranges[1], fineness)
		Y = np.arange(ranges[0], ranges[1], fineness)
		X, Y = np.meshgrid(X, Y)

		# determine pdf over grid
		Z = mixture_pdf(X, Y, self.pi, self.centroids, self.covmat)

		# plot the 3D surface
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(X, Y, Z, cmap=plt.get_cmap("coolwarm"), linewidth=0,
		                       antialiased=False, shade=True, alpha=0.7)
		ax.view_init(50, 100)
		fig.savefig(filename)
		return fig

	def contour_gm(self, ranges, fineness, filename, samples=None, labels=None):
		"""
		Surface plot of gaussian mixture
		:param mix_out: output of method pdf_gm
		:param range: 2 by 1 vector of plot range
		:param grid: fineness of plot mesh
		:param labels: variable labels
		:return: na
		"""
		if self.centroids.shape[1] > 2:
			print("Only bivariate data allowed for contour_gm.")

		def mixture_pdf(X, Y, pi, centroids, covmat):
			"""
			Bivariate pdf of gaussian mixture
			:param X: data vector for variable x
			:param Y: data vector for variable y
			:param pi: mixture parameter
			:param centroids: means for multivariate normal
			:param covmat: covariance matrix for multivariate normal
			:return: pdf of MVN
			"""
			# initialize counting params
			K = len(pi)
			D = centroids.shape[1]
			N = len(X)

			# sum over weighted bivariate normal pdf's
			mix = np.zeros([N, N])
			for k in range(K):
				mix = mix + pi[k] * mlab.bivariate_normal(X, Y, covmat[k, 0], covmat[k, 1], centroids[k, 0], centroids[k, 1], 0)
			return mix

		# create grid for 3D surface plot
		X = np.arange(ranges[0], ranges[1], fineness)
		Y = np.arange(ranges[0], ranges[1], fineness)
		X, Y = np.meshgrid(X, Y)

		# determine pdf over grid
		Z = mixture_pdf(X, Y, self.pi, self.centroids, self.covmat)

		# plot contour
		fig = plt.figure()
		ax = fig.gca()
		ax.contour(X, Y, Z)

		if samples != None:
			ax.scatter(samples[:, 0], samples[:, 1], s=1, color='0.5')

		fig.savefig(filename)
		return fig


	def pairs_gm(self, data, filename, ranges, labels=None):
		"""
		pair-wise contour plot of gaussian mixture
		:param mix: output of method pdf_gm
		:param labels: variable labels
		:return: na
		"""
		if data.shape[1] > 5:
			print("More than 5 variables with pairs_gm may take considerable time.")

		def mixture_pdf(X, Y, pi, centroids, covmat):
			"""
			Bivariate pdf of gaussian mixture
			:param X: data vector for variable x
			:param Y: data vector for variable y
			:param pi: mixture parameter
			:param centroids: means for multivariate normal
			:param covmat: covariance matrix for multivariate normal
			:return: pdf of MVN
			"""
			# initialize counting params
			K = len(pi)
			D = centroids.shape[1]
			N = len(X)

			# weighted sum of bivariate normal pdf's
			mix = np.zeros([N, N])
			for k in range(K):
				mix = mix + pi[k] * mlab.bivariate_normal(X, Y, covmat[k, 0], covmat[k, 1], centroids[k, 0], centroids[k, 1], 0)
			return mix

		# initialize number of variables
		n_variables = data.shape[1]

		# specify labels if any
		if labels is None:
			labels = ['var%d' % i for i in range(n_variables)]

		# initialize plot and iterate over each variable.
		# for each variable, perform a visualization over grid
		fig = plt.figure()
		for i in range(n_variables):
			for j in range(n_variables):
				# create sub-plots for a given variable
				n_sub = i * n_variables + j + 1
				ax = fig.add_subplot(n_variables, n_variables, n_sub)
				ax.axes.get_xaxis().set_ticklabels([])
				ax.axes.get_yaxis().set_ticklabels([])
				if i == j:
					# diagonal plots are histograms (i==j)
					ax.hist(data[:, i])
					ax.set_title(labels[i])
				else:
					# off-diagonal plots are scatter and contours
					X = np.arange(ranges[0], ranges[1], 0.1)
					Y = np.arange(ranges[0], ranges[1], 0.1)
					X, Y = np.meshgrid(X, Y)
					Z = mixture_pdf(X, Y, self.pi, self.centroids[:, (i, j)], self.covmat[:, (i, j)])
					ax.contour(X, Y, Z)
					ax.scatter(data[:, i], data[:, j], s=1, color='0.5')
		fig.savefig(filename)
		return fig

	def write_gm(self, x_out, filename):
		"""
		Write simulation output to file
		:param x_out: simulation output from simulate_gm
		:param filename: name of file
		:return: na
		"""
		np.savetxt(filename, x_out, delimiter=",")

	def write_params(self, filename):
		"""
		Write parameters to file
		:param filename: name of file
		:return: na
		"""
		true_moment1 = np.dot(self.pi, self.centroids)
		true_moment2 = np.dot(np.power(self.pi, 2), self.centroids)
		np.savetxt(filename + "_TrueFirstMoment.csv", true_moment1, delimiter=",")
		np.savetxt(filename + "_TrueSecondMoment.csv", true_moment2, delimiter=",")
		np.savetxt(filename + "_TruePi.csv", self.pi, delimiter=",")
		np.savetxt(filename + "_TrueCentroids.csv", self.centroids, delimiter=",")
		np.savetxt(filename + "_TrueCov.csv", self.covmat, delimiter=",")


