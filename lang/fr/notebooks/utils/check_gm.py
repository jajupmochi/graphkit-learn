#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check basic properties of gram matrices.
Created on Wed Sep 19 15:32:29 2018

@author: ljia
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

# read gram matrices from file.
results_dir = '../results/marginalizedkernel/myria'
ds_name = 'ENZYMES'
gmfile = np.load(results_dir + '/' + ds_name + '.gm.npz')
#print('gm time: ', gmfile['gmtime'])
# a list to store gram matrices for all param_grid_precomputed
gram_matrices = gmfile['gms']
# param_list_pre_revised = gmfile['params'] # list to store param grids precomputed ignoring the useless ones
#y = gmfile['y'].tolist()
#x = gram_matrices[0]

for idx, x in enumerate(gram_matrices):
    print()
    print(idx)
    plt.imshow(x)
    plt.colorbar()
    plt.savefig('../check_gm/' + ds_name + '.gm.eps', format='eps', dpi=300)
#    print(np.transpose(x))
    print('if symmetric: ', np.array_equal(x, np.transpose(x)))
    
    print('diag: ', np.diag(x))
    print('sum diag < 0.1: ', np.sum(np.diag(x) < 0.1))
    print('min, max diag: ', min(np.diag(x)), max(np.diag(x)))
    print('min, max matrix: ', np.min(x), np.max(x))
    for i in range(len(x)):
        for j in range(len(x)):
            if x[i][j] > 1 + 1e-9:
                print(i, j)
                raise Exception('value bigger than 1 with index', i, j)
    print('mean x: ', np.mean(np.mean(x)))
    
    [lamnda, v] = eig(x)
    print('min, max lambda: ', min(lamnda), max(lamnda))
    if -1e-10 > min(lamnda):
        raise Exception('wrong eigen values.')
