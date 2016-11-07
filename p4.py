import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from collections import namedtuple

train = pd.read_csv('./mnist_train.csv', header=None)
X = train.iloc[:, 1:].as_matrix()
X = (X > X[X > 0].mean()).astype(float)
y = train.iloc[:, 0].as_matrix()
del train

def EM(X, k, theta, objective, likelihood, m_step, max_iter=100, print_freq=10):
	r = np.ones((X.shape[0], k)) / k
	pi = np.ones(k) / k
	objectives = [objective(X, r, pi, theta)]

	for i in range(max_iter):
		if (i % print_freq) == 0:
			print("[i={}] objective={}".format(i, objectives[-1]))

		# E step: approximate r
		r = likelihood(X, theta) * pi
		r = r / r.sum(axis=1)[:, np.newaxis]
		# M step: maximize using approximation
		pi, theta = m_step(X, r)

		objectives.append(objective(X, r, pi, theta))

	return (objectives, r, pi, theta)

def bernoullis(X, k, prior_dir_a, prior_beta_a, prior_beta_b, max_iter=50, print_freq=10):
	S_0 = np.diag(np.std(X, axis=0) ** 2) / k ** (1 / X.shape[1])
	Theta = namedtuple('BMM', 'mean')

	theta = Theta(mean=X[:k*np.floor(X.shape[0] / k)].reshape(k, -1, X.shape[1],).mean(axis=1))

	def likelihood(X, theta):
		p = np.tile(theta.mean.T, (X.shape[0], 1, 1))
		p[X == 0] = 1 - p[X == 0]
		p = p.prod(axis=1)
		return p

	denom = X.shape[0] + prior_dir_a.sum() - k
	def m_step(X, r):
		r_sum = r.sum(axis=0)
		pi = (r_sum + prior_dir_a - 1) / denom
		mu = (((X[:,:,np.newaxis] * r[:, np.newaxis, :]).sum(axis=0) + prior_beta_a - 1) / \
			(r_sum + prior_beta_a + prior_beta_b - 2))
		mu = mu.transpose()
		return pi, Theta(mean=mu)
	
	def objective(X, r, pi, theta):
		log_prior = np.log(scipy.stats.beta.pdf(
			theta.mean, prior_beta_a, prior_beta_b
		)).sum() + np.log(scipy.stats.dirichlet.pdf(pi, alpha=prior_alpha))
		pi_term = (r * np.log(pi)[np.newaxis, :]).sum()
		likelihood_term = r * np.log(likelihood(X, theta))
		likelihood_term_sum = likelihood_term[r > 1e-12].sum()
		return likelihood_term_sum + pi_term + log_prior

	return EM(X, k, theta, objective, likelihood, m_step, max_iter=max_iter, print_freq=print_freq,)

np.random.seed(1)
N = int(10000)
subset_ix = np.random.randint(0, X.shape[0], (N,))
smallX = X[subset_ix]

k = 10
obj, r, pi, theta = bernoullis(smallX, k, prior_dir_a=np.ones(10), \
	prior_beta_a=1, prior_beta_b=1, max_iter=50, print_freq=10)

# Convergence plot
plt.plot(obj)
plt.style.use('bmh')
plt.title('MAP Bernoullis Mixture')
plt.xlabel('Iterations')
plt.ylabel('Log Likelohood')
plt.tight_layout()
# plt.show()

# Mean Plot
plt.figure(figsize=(5,2))
for i in range(10):
	plt.subplot(2, 5, i + 1)
	img = theta.mean[i]
	plt.imshow(img.reshape(28, 28), cmap='Greys')
	plt.axis('off')

plt.suptitle('Means of Bernoullis Mixture')
plt.tight_layout()
# plt.show()
