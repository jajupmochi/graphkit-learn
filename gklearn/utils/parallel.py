#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:39:46 2018
Parallel aid functions.
@author: ljia
"""
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import sys

def parallel_me(func, func_assign, var_to_assign, itr, len_itr=None, init_worker=None, 
                glbv=None, method=None, n_jobs=None, chunksize=None, itr_desc='',
                verbose=True):
    '''
    '''
    if method == 'imap_unordered':
        if glbv: # global varibles required.
#            def init_worker(v_share):
#                global G_var
#                G_var = v_share
            if n_jobs == None:
                n_jobs = multiprocessing.cpu_count()
            with Pool(processes=n_jobs, initializer=init_worker, 
                      initargs=glbv) as pool:                
                if chunksize == None:
                    if len_itr < 100 * n_jobs:
                        chunksize = int(len_itr / n_jobs) + 1
                    else:
                        chunksize = 100
                for result in (tqdm(pool.imap_unordered(func, itr, chunksize),
                    desc=itr_desc, file=sys.stdout) if verbose else 
                    pool.imap_unordered(func, itr, chunksize)):
                    func_assign(result, var_to_assign)
        else:
            if n_jobs == None:
                n_jobs = multiprocessing.cpu_count()
            with Pool(processes=n_jobs) as pool:
                if chunksize == None:
                    if len_itr < 100 * n_jobs:
                        chunksize = int(len_itr / n_jobs) + 1
                    else:
                        chunksize = 100
                for result in (tqdm(pool.imap_unordered(func, itr, chunksize),
                    desc=itr_desc, file=sys.stdout) if verbose else 
                    pool.imap_unordered(func, itr, chunksize)):
                    func_assign(result, var_to_assign)
            
                    

def parallel_gm(func, Kmatrix, Gn, init_worker=None, glbv=None, 
                method='imap_unordered', n_jobs=None, chunksize=None, 
                verbose=True):
    from itertools import combinations_with_replacement
    def func_assign(result, var_to_assign):
        var_to_assign[result[0]][result[1]] = result[2]
        var_to_assign[result[1]][result[0]] = result[2]
    itr = combinations_with_replacement(range(0, len(Gn)), 2)
    len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
    parallel_me(func, func_assign, Kmatrix, itr, len_itr=len_itr,
        init_worker=init_worker, glbv=glbv, method=method, n_jobs=n_jobs, 
        chunksize=chunksize, itr_desc='calculating kernels', verbose=verbose)