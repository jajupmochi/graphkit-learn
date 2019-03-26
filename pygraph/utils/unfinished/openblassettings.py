#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:31:01 2018
A script to set the thread number of OpenBLAS (if used). 
Some modules (such as Numpy, Scipy, sklearn) using OpenBLAS perform parallel 
computation automatically, which causes conflict when other paralleling modules 
such as multiprossing.Pool, highly increase the computing time. By setting 
thread to 1, OpenBLAS is forced to use single thread/CPU, thus this conflict 
can be avoided.
-e.g:
    with num_threads(8):
    np.dot(x, y)
@author: ali_m
@Reference: ali_m, https://stackoverflow.com/a/29582987, 2018.12
"""

import contextlib
import ctypes
from ctypes.util import find_library
import os

# Prioritize hand-compiled OpenBLAS library over version in /usr/lib/
# from Ubuntu repos
try_paths = ['/opt/OpenBLAS/lib/libopenblas.so',
             '/lib/libopenblas.so',
             '/usr/lib/libopenblas.so.0',
             find_library('openblas')]
openblas_lib = None
for libpath in try_paths:
    try:
        openblas_lib = ctypes.cdll.LoadLibrary(libpath)
        break
    except OSError:
        continue
if openblas_lib is None:
    raise EnvironmentError('Could not locate an OpenBLAS shared library', 2)


def set_num_threads(n):
    """Set the current number of threads used by the OpenBLAS server."""
    openblas_lib.openblas_set_num_threads(int(n))


# At the time of writing these symbols were very new:
# https://github.com/xianyi/OpenBLAS/commit/65a847c
try:
    openblas_lib.openblas_get_num_threads()
    def get_num_threads():
        """Get the current number of threads used by the OpenBLAS server."""
        return openblas_lib.openblas_get_num_threads()
except AttributeError:
    def get_num_threads():
        """Dummy function (symbol not present in %s), returns -1."""
        return -1
    pass

try:
    len(os.sched_getaffinity(0))
    def get_num_procs():
        """Get the total number of physical processors"""
        return len(os.sched_getaffinity(0))
except AttributeError:
    def get_num_procs():
        """Dummy function (symbol not present), returns -1."""
        return -1
    pass


@contextlib.contextmanager
def num_threads(n):
    """Temporarily changes the number of OpenBLAS threads.

    Example usage:

        print("Before: {}".format(get_num_threads()))
        with num_threads(n):
            print("In thread context: {}".format(get_num_threads()))
        print("After: {}".format(get_num_threads()))
    """
    old_n = get_num_threads()
    set_num_threads(n)
    try:
        yield
    finally:
        set_num_threads(old_n)