from ctypes import *
import os
lib1 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/fann/libdoublefann.so')
lib2 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/libsvm.3.22/libsvm.so')
lib3 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/nomad/libnomad.so')
lib4 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/nomad/libsgtelib.so')
