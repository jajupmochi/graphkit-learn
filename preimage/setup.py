#from distutils.core import setup
from distutils.extension import Extension
#from Cython.Distutils import build_ext

from distutils.core import setup
from Cython.Build import cythonize

#setup(ext_modules=cythonize("script.pyx"))

extensions = [Extension("script",
                        sources=["script.pyx", "src/essai.cpp"],
                        include_dirs=["include","include/lsape", "include/Eigen", "include/nomad", "include/sgtelib", "include/libsvm.3.22", "include/fann", "include/boost_1_69_0"],
                        library_dirs=["lib/fann","lib/gedlib", "lib/libsvm.3.22","lib/nomad"],
                        libraries=["doublefann","sgtelib", "svm", "nomad"],
                        language="c++",
                        extra_compile_args=["-std=c++11"],
                        extra_link_args=["-std=c++11"])]

setup(ext_modules=cythonize(extensions))

#extensions = [Extension("script", sources=["script.pyx", "include/gedlib-master/src/env/ged_env.ipp"],  include_dirs=["."], language="c++")]
 
#setup(name = "script", ext_modules = extensions, cmdclass = {'build_ext':build_ext},)


# Commande Bash : python setup.py build_ext --inplace
