#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:40:52 2018

@author: ljia
"""

import functools
from libs import *
import multiprocessing

from gklearn.kernels.ssp_sym import structuralspkernel
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct

dslist = [
    {'name': 'Letter-med', 'dataset': '../datasets/Letter-med/Letter-med_A.txt'},
    # node nsymb
    {'name': 'ENZYMES', 'dataset': '../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'},
    # node symb/nsymb
]
estimator = structuralspkernel
mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
param_grid_precomputed = {'node_kernels': 
    [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}],
    'edge_kernels': 
    [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}]}
param_grid = [{'C': np.logspace(-10, 10, num=41, base=10)},
              {'alpha': np.logspace(-10, 10, num=41, base=10)}]

for ds in dslist:
    print()
    print(ds['name'])
    model_selection_for_precomputed_kernel(
        ds['dataset'],
        estimator,
        param_grid_precomputed,
        (param_grid[1] if ('task' in ds and ds['task']
                           == 'regression') else param_grid[0]),
        (ds['task'] if 'task' in ds else 'classification'),
        NUM_TRIALS=30,
        datafile_y=(ds['dataset_y'] if 'dataset_y' in ds else None),
        extra_params=(ds['extra_params'] if 'extra_params' in ds else None),
        ds_name=ds['name'],
        n_jobs=multiprocessing.cpu_count(),
        read_gm_from_file=False)
    print()