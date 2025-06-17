from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize

import os
import shutil

# clean previous build
for name in os.listdir():
    if (name.startswith('gedlibpy') and not(name.endswith('.pyx') or name.endswith('.pxd'))):
        os.remove(name)
    if name == 'build':
        shutil.rmtree(name)
        

extensions = [Extension("gedlibpy",
                        # sources=["gedlibpy.pyx", "src/GedLibBind.cpp"],
                        sources=["gedlibpy.pyx"],
                        include_dirs=["src", "include", "include/lsape", "include/Eigen", "include/nomad", "include/sgtelib", "include/libsvm.3.22", "include/fann", "include/boost_1_69_0"],
                        library_dirs=["lib/fann", "lib/gedlib", "lib/libsvm.3.22","lib/nomad"],
                        libraries=["doublefann", "sgtelib", "svm", "nomad"],
                        # library_dirs=["."],
                        # libraries=["gxlgedlib"],
                        language="c++",
                        extra_compile_args=["-std=c++11"],
                        extra_link_args=["-std=c++11"])]

setup(ext_modules=cythonize(extensions,
                            compiler_directives={'language_level': '3'}))
# setup(ext_modules=cythonize(extensions))


# Commande Bash : python setup.py build_ext --inplace
