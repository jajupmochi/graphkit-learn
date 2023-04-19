"""Those who are not graph kernels. We can be kernels for nodes or edges!
These kernels are defined between pairs of vectors.
"""
import numpy as np


def kronecker_delta_kernel(x, y):
	"""Delta kernel. Return 1 if x == y, 0 otherwise.

	Parameters
	----------
	x, y : any
		Two parts to compare.

	Return
	------
	kernel : integer
		Delta kernel.

	References
	----------
	[1] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between
	labeled graphs. In Proceedings of the 20th International Conference on
	Machine Learning, Washington, DC, United States, 2003.
	"""
	return (1 if np.array_equal(x, y) else 0)


def delta_kernel(x, y):
	return x == y  # (1 if condition else 0)


def deltakernel(x, y):
	return delta_kernel(x, y)


def gaussian_kernel(x, y, gamma=None):
	"""Gaussian kernel.
	Compute the rbf (gaussian) kernel between x and y:

		K(x, y) = exp(-gamma ||x-y||^2).

	Read more in the `User Guide of scikit-learn library <https://scikit-learn.org/stable/modules/metrics.html#rbf-kernel>`__.

	Parameters
	----------
	x, y : array

	gamma : float, default None
		If None, defaults to 1.0 / n_features

	Returns
	-------
	kernel : float
	"""
	if gamma is None:
		gamma = 1.0 / len(x)

	# xt = np.array([float(itm) for itm in x]) # @todo: move this to dataset or datafile to speed up.
	# yt = np.array([float(itm) for itm in y])
	# 	kernel = xt - yt
	# 	kernel = kernel ** 2
	# 	kernel = np.sum(kernel)
	# 	kernel *= -gamma
	# 	kernel = np.exp(kernel)
	# 	return kernel

	return np.exp((np.sum(np.subtract(x, y) ** 2)) * -gamma)


def tanimoto_kernel(x, y):
	xy = np.dot(x, y)
	return xy / (np.dot(x, x) + np.dot(y, y) - xy)


def gaussiankernel(x, y, gamma=None):
	return gaussian_kernel(x, y, gamma=gamma)


def polynomial_kernel(x, y, gamma=1, coef0=0, d=1):
	return (np.dot(x, y) * gamma + coef0) ** d


def highest_polynomial_kernel(x, y, d=1, c=0):
	"""Polynomial kernel.
	Compute the polynomial kernel between x and y:

		K(x, y) = <x, y> ^d + c.

	Parameters
	----------
	x, y : array

	d : integer, default 1

	c : float, default 0

	Returns
	-------
	kernel : float
	"""
	return np.dot(x, y) ** d + c


def polynomialkernel(x, y, d=1, c=0):
	return highest_polynomial_kernel(x, y, d=d, c=c)


def linear_kernel(x, y):
	"""Polynomial kernel.
	Compute the polynomial kernel between x and y:

		K(x, y) = <x, y>.

	Parameters
	----------
	x, y : array

	d : integer, default 1

	c : float, default 0

	Returns
	-------
	kernel : float
	"""
	return np.dot(x, y)


def linearkernel(x, y):
	return linear_kernel(x, y)


def cosine_kernel(x, y):
	return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def sigmoid_kernel(x, y, gamma=None, coef0=1):
	if gamma is None:
		gamma = 1.0 / len(x)

	k = np.dot(x, y)
	k *= gamma
	k += coef0
	k = np.tanh(k)
	# 	k = np.tanh(k, k) # compute tanh in-place
	return k


def laplacian_kernel(x, y, gamma=None):
	if gamma is None:
		gamma = 1.0 / len(x)

	k = -gamma * np.linalg.norm(np.subtract(x, y))
	k = np.exp(k)
	return k


def chi2_kernel(x, y, gamma=1.0):
	k = np.divide(np.subtract(x, y) ** 2, np.add(x, y))
	k = np.sum(k)
	k *= -gamma
	return np.exp(k)


def exponential_kernel(x, y, gamma=None):
	if gamma is None:
		gamma = 1.0 / len(x)

	return np.exp(np.dot(x, y) * gamma)


def intersection_kernel(x, y):
	return np.sum(np.minimum(x, y))


def multiquadratic_kernel(x, y, c=0):
	return np.sqrt((np.sum(np.subtract(x, y) ** 2)) + c)


def inverse_multiquadratic_kernel(x, y, c=0):
	return 1 / multiquadratic_kernel(x, y, c=c)


def kernelsum(k1, k2, d11, d12, d21=None, d22=None, lamda1=1, lamda2=1):
	"""Sum of a pair of kernels.

	k = lamda1 * k1(d11, d12) + lamda2 * k2(d21, d22)

	Parameters
	----------
	k1, k2 : function
		A pair of kernel functions.
	d11, d12:
		Inputs of k1. If d21 or d22 is None, apply d11, d12 to both k1 and k2.
	d21, d22:
		Inputs of k2.
	lamda1, lamda2: float
		Coefficients of the product.

	Return
	------
	kernel : integer

	"""
	if d21 is None or d22 is None:
		kernel = lamda1 * k1(d11, d12) + lamda2 * k2(d11, d12)
	else:
		kernel = lamda1 * k1(d11, d12) + lamda2 * k2(d21, d22)
	return kernel


def kernelproduct(k1, k2, d11, d12, d21=None, d22=None, lamda=1):
	"""Product of a pair of kernels.

	k = lamda * k1(d11, d12) * k2(d21, d22)

	Parameters
	----------
	k1, k2 : function
		A pair of kernel functions.
	d11, d12:
		Inputs of k1. If d21 or d22 is None, apply d11, d12 to both k1 and k2.
	d21, d22:
		Inputs of k2.
	lamda: float
		Coefficient of the product.

	Return
	------
	kernel : integer
	"""
	if d21 is None or d22 is None:
		kernel = lamda * k1(d11, d12) * k2(d11, d12)
	else:
		kernel = lamda * k1(d11, d12) * k2(d21, d22)
	return kernel


if __name__ == '__main__':
	o = polynomialkernel([1, 2], [3, 4], 2, 3)
