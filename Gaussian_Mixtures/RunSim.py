import numpy as np
import matplotlib.pyplot as plt
from GaussianMixture import GaussianMixture


# ********************************************
# inpute parameters below
# ********************************************
pi = np.array([20, 34, 32, 45, 40, 29, 70])
pi = pi/pi.sum()

centroids = np.array([[0, 0],
					 [-15, 5],
					 [-5, -9],
					 [4, -3],
					 [8, 12],
					 [2, 2],
					 [15, -1]])

covmat = np.array([[3, 4],
					 [3, 2.5],
					 [2, 3],
					 [14, 2],
					 [4, 3],
					 [3, 4],
					 [4, 13]])

sim2d_wgan = np.load("wgan_2gaussians2d_3dnoise.npy")
sim2d_wgan_2 = np.load("wgan_2gaussians2d_3dnoise_0.9sdev.npy")
sim2d_vgan = np.load("vgan_2gaussians2d_3dnoise_0.9sdev.npy")
sim2d_vgan_2 = np.load("vgan_2gaussians2d_3dnoise_0.9sddev.npy")
sim2d_vgan_3 = np.load("vgan_2gaussians2d_3dnoise.npy")

sim2d_vgan_2[:,0] =  sim2d_vgan_2[:,0] - 8
sim2d_vgan_2[:,1] =  sim2d_vgan_2[:,1] + 4

sim2d_vgan[:,1] =  sim2d_vgan[:,1] + 2.5

sim2d_vgan_3[:,1] = sim2d_vgan_3[:,1] + 2.5


pi = np.array([0.5, 0.5])
centroids = np.array([[-1, 0], [1, 0]])
covmat = np.array([[0.6, 0.6], [0.6, 0.6]])

# ********************************************
# Initialize GaussianMixture class
# ********************************************
gm = GaussianMixture(pi, centroids, covmat)

# ********************************************
# Simulate from Gaussian mixture
# ********************************************
gm_sim = gm.simulate_gm(n_samples=100)

# ********************************************
# Obtain pdf of mixture for some input X
# ********************************************
pdf = gm.pdf_gm(gm_sim)


gm.contour_gm(ranges=[-25,25], fineness=0.1, samples=gm_sim, filename="contour_3.png")

# ********************************************
# Visualize pdf of mixture in bivariate case
# ********************************************
gm.surface_gm(ranges=[-25,25], fineness=.1, filename="surfaceplot.png")


# ********************************************
# Visualize pdf of mixture in bivariate case
# ********************************************
gm.pairs_gm(gm_sim, filename="pairsplot.png", ranges=[-25,25])
