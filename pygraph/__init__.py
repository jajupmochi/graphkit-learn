# -*-coding:utf-8 -*-
"""
Pygraph

This  package contains 4 sub  packages :
        * c_ext : binders to C++ code
        * ged : allows to compute graph edit distance between networkX graphs
        * kernels : computation of graph kernels, ie graph similarity measure compatible with SVM
        * notebooks : examples of code using this library
        * utils : Diverse computation on graphs
"""

# info
__version__ = "0.1"
__author__  = "Benoit Gaüzère"
__date__    = "November 2017"
 
# import sub modules
# from pygraph import c_ext
# from pygraph import ged
from pygraph import utils
