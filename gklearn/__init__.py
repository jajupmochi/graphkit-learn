# -*-coding:utf-8 -*-
"""
gklearn

This  package contains 4 sub  packages :
        * c_ext : binders to C++ code
        * ged : allows to compute graph edit distance between networkX graphs
        * kernels : computation of graph kernels, ie graph similarity measure compatible with SVM
        * notebooks : examples of code using this library
        * utils : Diverse computation on graphs
"""

# info
import datetime

__version__ = '0.2.1.post' + datetime.now().strftime('%Y%m%d%H%M%S')
__author__ = 'Linlin Jia, Benoit Gaüzère, Paul Honeine'
__date__ = 'November 2017'

# import sub modules
# from gklearn import c_ext
# from gklearn import ged
# from gklearn import utils
