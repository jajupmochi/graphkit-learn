import importlib
import os
import shutil
import sys
from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize


def install(package):
	try:
		importlib.import_module(package)
	except ImportError:
		import subprocess
		cur_python = sys.executable
		subprocess.call([cur_python, '-m', 'pip', 'install', package])


# def install_gedlib():
# 	"""
# 	Notes
# 	-----
# 	The followings are done during this installation:
# 		- Build files in `ext/fann.2.2.0/lib/`;
# 		- Clean obj files in `ext/nomad.3.8.1/ext/sgtelib/src/`;
# 		- Build files in `ext/nomad.3.8.1/ext/sgtelib/lib/`;
# 		- Generate executable files in `ext/nomad.3.8.1/ext/sgtelib/bin/`;
# 		- Build files in `ext/nomad.3.8.1/lib/`;
# 		- Generate `ext/libsvm.3.22/libsvm.so`;
# 	"""
# 	print()
# 	print('Installing GEDLIB...')
# 	import subprocess
# 	cur_python = sys.executable
# 	subprocess.call([cur_python, '--version'])
# 	subprocess.call(['which', cur_python])
# 	gedlib_dir = 'include/gedlib-master/'
# 	subprocess.call(
# 		[cur_python, 'install.py'], cwd=gedlib_dir
# 	)
# 	print()


# def clean_previous_build():
# 	# clean previous build
# 	print()
# 	print('Cleaning previous build...')
# 	for name in os.listdir():
# 		if (name.startswith('gedlibpy') and not (
# 				name.endswith('.pyx') or name.endswith('.pxd'))):
# 			os.remove(name)
# 		if name == 'build':
# 			shutil.rmtree(name)
# 	print('Done!')
# 	print()


# def get_extensions(include_glibc):
def get_extensions():
	exts = [
		# Extension(
		# 	"gedlibpy",
		# 	# sources=["gedlibpy.pyx", "src/GedLibBind.cpp"],
		# 	sources=[
		# 		"common_bind.pyx", "gedlibpy_gxl.pyx", "gedlibpy_attr.pyx",
		# 		"src/gedlib_bind_gxl.cpp", "src/gedlib_bind_attr.cpp", "src/gedlib_bind_util.cpp",
		# 		# "include/gedlib-master/src/env/ged_env.gxl.cpp"
		# 	],
		# 	include_dirs=[
		# 		"src",
		# 		"include",
		# 		"include/lsape.5/include",
		# 		"include/eigen.3.3.4/Eigen",
		# 		"include/nomad.3.8.1/src",
		# 		"include/nomad.3.8.1/ext/sgtelib/src",
		# 		"include/libsvm.3.22",
		# 		"include/fann.2.2.0/include",
		# 		"include/boost.1.69.0"
		# 	],
		# 	library_dirs=[
		# 		"lib/fann.2.2.0", "lib/libsvm.3.22",  # "lib/gedlib",
		# 		"lib/nomad.3.8.1"
		# 	],
		# 	libraries=["doublefann", "sgtelib", "svm", "nomad"],
		# 	language="c++",
		# 	extra_compile_args=["-std=c++17"],  # , "-DGXL_GEDLIB_SHARED"],
		# 	extra_link_args=["-std=c++17"]
		# )
		Extension(
			"gedlibpy_gxl",
			# sources=["gedlibpy.pyx", "src/GedLibBind.cpp"],
			sources=[
				# "include/gedlib-master/src/env/ged_env.gxl.cpp",
				"gedlibpy_gxl.pyx",
				# "src/gedlib_bind_gxl.cpp",
				# "src/gedlib_bind_util.cpp",
			],
			include_dirs=[
				"src",
				"include",
				"include/lsape.5/include",
				"include/eigen.3.3.4/Eigen",
				"include/nomad.3.8.1/src",
				"include/nomad.3.8.1/ext/sgtelib/src",
				"include/libsvm.3.22",
				"include/fann.2.2.0/include",
				"include/boost.1.69.0"
			],
			library_dirs=[
				"lib/fann.2.2.0", "lib/libsvm.3.22",  # "lib/gedlib",
				"lib/nomad.3.8.1"
			],
			libraries=["doublefann", "sgtelib", "svm", "nomad"],
			language="c++",
			extra_compile_args=["-std=c++17"],  # , "-DGXL_GEDLIB_SHARED"],
			extra_link_args=["-std=c++17"]
		),
		Extension(
			"gedlibpy_attr",
			# sources=["gedlibpy.pyx", "src/GedLibBind.cpp"],
			sources=[
				# "include/gedlib-master/src/env/ged_env.gxl.cpp",
				"gedlibpy_attr.pyx",
				# "src/gedlib_bind_gxl.cpp",
				# "src/gedlib_bind_util.cpp",
			],
			include_dirs=[
				"src",
				"include",
				"include/lsape.5/include",
				"include/eigen.3.3.4/Eigen",
				"include/nomad.3.8.1/src",
				"include/nomad.3.8.1/ext/sgtelib/src",
				"include/libsvm.3.22",
				"include/fann.2.2.0/include",
				"include/boost.1.69.0"
			],
			library_dirs=[
				"lib/fann.2.2.0", "lib/libsvm.3.22",  # "lib/gedlib",
				"lib/nomad.3.8.1"
			],
			libraries=["doublefann", "sgtelib", "svm", "nomad"],
			language="c++",
			extra_compile_args=["-std=c++17"],  # , "-DGXL_GEDLIB_SHARED"],
			extra_link_args=["-std=c++17"]
		),
	]
	return exts


def remove_includes():
	print()
	print('Deleting includes...')
	name = os.path.join(os.getcwd(), 'include/')
	shutil.rmtree(name)
	print('Done!')
	print()


if __name__ == '__main__':
	# if args.build_gedlib == 'true':
	# 	# Install GEDLIB:
	# 	install_gedlib()

	# # clean previous build:
	# clean_previous_build()

	print()
	print('Start building...')
	# Build gedlibpy:
	# extensions = get_extensions(include_glibc)
	extensions = get_extensions()
	with open("README.rst", "r") as fh:
		long_description = fh.read()

	# Attention: setup function can not be put inside a function!
	setup(
		ext_modules=cythonize(
			extensions,
			compiler_directives={'language_level': '3'},
			# Generate .html files for Cython annotations, should be set to False for production
			# (i.e., when the package is installed, not when it is developed):
			annotate=True,  # todo
			# Only recompile if the .pyx files are modified:
			force=True,  # fixme: check if it still works if c++ wrappers are modified
			# Use N threads for compilation multiple .pyx files to .cpp files (works
			# only if there are multiple extensions):
			nthreads=4,  # todo: change as needed
		),
		name="gedlibpy",
		author="Lambert Natacha and Linlin Jia",
		author_email="jajupmochi@gmail.com",
		description="A Python wrapper library for C++ library GEDLIB of graph edit distances",
		long_description=long_description,
		long_description_content_type="text/markdown",
		project_urls={
			# 'Documentation': 'https://graphkit-learn.readthedocs.io',
			'Source': 'https://github.com/jajupmochi/graphkit-learn/tree/master/gklearn/gedlib',
			'Tracker': 'https://github.com/jajupmochi/graphkit-learn/issues',
		},
		url="https://github.com/jajupmochi/graphkit-learn/tree/master/gklearn/gedlib",
		license="GPL-3.0-or-later",  # SPDX license identifier
		classifiers=[
			"Programming Language :: Python :: 3",
			# "License :: OSI Approved",
			"Operating System :: OS Independent",
			'Intended Audience :: Science/Research',
			'Intended Audience :: Developers',
		],
		include_dirs=[numpy.get_include()],
		zip_safe=False,  # Do not zip the package, so that .so files can be loaded correctly
	)

	# List generated files:
	print()
	print('The following files are generated:')
	for name in os.listdir():
		if (name.startswith('gedlibpy') and not (
				name.endswith('.pyx') or name.endswith('.pxd'))):
			print(name)

	print()
	print('Build completed!')
	print()

# Bash Command:
# python3 setup_core.py build_ext --inplace --parallel 8
# -- Explain:
# -- --parallel N: parallel the construction of .cpp files using c++ compiler (g++ or
#    clang++) with N threads (on c++ layer, from .cpp to .so).
#    It is different from the N threads in cythonize(), which is on cython layer (from
#    .pyx to .cpp).
# if error: command 'clang++' failed: No such file or directory:
# Add "CXX=g++". e.g.,
# CXX=g++ python setup_core.py build_ext --inplace


