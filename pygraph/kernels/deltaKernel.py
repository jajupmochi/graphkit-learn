def deltakernel(condition):
    """Return 1 if condition holds, 0 otherwise.
    
    Parameters
    ----------
    condition : Boolean
        A condition, according to which the kernel is set to 1 or 0.
        
    Return
    ------
    Kernel : integer
        Delta Kernel.
        
    References
    ----------
    [1] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between labeled graphs. In Proceedings of the 20th International Conference on Machine Learning, Washington, DC, United States, 2003.
    """
    return (1 if condition else 0)