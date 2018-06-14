"""Those who are not graph kernels. We can be kernels for nodes or edges!
"""


def deltakernel(x, y):
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
    [1] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between labeled graphs. In Proceedings of the 20th International Conference on Machine Learning, Washington, DC, United States, 2003.
    """
    return x == y  #(1 if condition else 0)


def gaussiankernel(x, y):
    """Gaussian kernel. Use sklearn.metrics.pairwise.rbf_kernel instead.
    """
    pass


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
    if d21 == None or d22 == None:
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
    if d21 == None or d22 == None:
        kernel = lamda * k1(d11, d12) * k2(d11, d12)
    else:
        kernel = lamda * k1(d11, d12) * k2(d21, d22)
    return kernel
