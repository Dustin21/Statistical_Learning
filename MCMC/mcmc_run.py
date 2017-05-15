import numpy as np
import matplotlib.pyplot as plt
from MCMC.mcmc import MCMC, DataDistribution


# ********************************************
# inpute parameters below
# ********************************************
mu1 = [-2.0, 2.0, -5.0, -10.0, 5]
sd1 = [1.0, 0.5, 0.5, 0.5, 1.5]
w1 = [0.2, 0.3, 0.2, 0.1, 0.2]
mixtureDens1 = DataDistribution(mu1, sd1, w1)


# ********************************************
# Initialize values for simulation:
# x0: starting point
# sigma: proposal distribution variance
# n: number of samples
# ********************************************
x0 = np.random.rand(1)
sigma = 10
n = int(1e5)

# ********************************************
# Run MCMC
# ********************************************
sim = MCMC(x0=x0, sigma=sigma, n=n)
x_out = sim.mcmc(mixtureDens1.gaussian_mixture)

# ********************************************
# Plot approximate density and true pdf
# ********************************************
sim.plotHistorgram(mixtureDens1.gaussian_mixture, x_out, xrange = [-15, 15])

# ********************************************
# Plot convergence of first moment
# ********************************************
sim.plotFirstMomentConvergence(x_out)
plt.axhline(np.dot(mu1,w1), color='g')
plt.show()

# ********************************************
# Plot convergence of standard deviation
# ********************************************
out = sim.plotSecondMomentConvergence(x_out)
#plt.axhline(np.dot(mu1,np.power(w1,2)), color='g')
plt.show()

