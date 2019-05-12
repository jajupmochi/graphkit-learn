from ctypes import *
lib1 = cdll.LoadLibrary('lib/fann/libdoublefann.so')
lib2 = cdll.LoadLibrary('lib/libsvm.3.22/libsvm.so')
lib3 = cdll.LoadLibrary('lib/nomad/libnomad.so')
lib4 = cdll.LoadLibrary('lib/nomad/libsgtelib.so')
