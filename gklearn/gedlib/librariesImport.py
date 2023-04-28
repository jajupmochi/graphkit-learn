from ctypes import *
import os
lib1 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/fann.2.2.0/libdoublefann.so')
lib2 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/libsvm.3.22/libsvm.so')
lib3 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/nomad.3.8.1/libnomad.so')
lib4 = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib/nomad.3.8.1/libsgtelib.so')
