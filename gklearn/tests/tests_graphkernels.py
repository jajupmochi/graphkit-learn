"""Tests of graph kernels.
"""

#import pytest
from gklearn.utils.graphfiles import loadDataset


def test_spkernel():
    """Test shortest path kernel.
    """
    from gklearn.kernels.spKernel import spkernel
    from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct
    import functools
    import multiprocessing

    ds_file = '../../datasets/Alkane/dataset.ds'
    ds_y = '../../datasets/Alkane/dataset_boiling_point_names.txt'
    Gn, y = loadDataset(ds_file, filename_y=ds_y)
    Gn = Gn[0:10]
    y = y[0:10]
    
    mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
    try:
        Kmatrix, run_time, idx = spkernel(Gn, node_label=None, node_kernels=
            {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel},
            n_jobs=multiprocessing.cpu_count(), verbose=True)
    except Exception as exception:
        assert False, exception
    

if __name__ == "__main__":
    test_spkernel()