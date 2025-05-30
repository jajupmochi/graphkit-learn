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
		Extension(
			"gedlibpy",
			# sources=["gedlibpy.pyx", "src/GedLibBind.cpp"],
			sources=["gedlibpy.pyx", "gedlibpy_attr.pyx"],
			include_dirs=[
				"src", "include", "include/lsape.5/include",
				"include/eigen.3.3.4/Eigen",
				"include/nomad.3.8.1/src",
				"include/nomad.3.8.1/ext/sgtelib/src",
				"include/libsvm.3.22",
				"include/fann.2.2.0/include", "include/boost.1.69.0"
			],
			library_dirs=[
				"lib/fann.2.2.0", "lib/libsvm.3.22",  # "lib/gedlib",
				"lib/nomad.3.8.1"
			],
			libraries=["doublefann", "sgtelib", "svm", "nomad"],
			language="c++",
			extra_compile_args=["-std=c++17"],
			extra_link_args=["-std=c++17"]
		)
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
			compiler_directives={'language_level': '3'}
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
		classifiers=[
			"Programming Language :: Python :: 3",
			"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
			"Operating System :: OS Independent",
			'Intended Audience :: Science/Research',
			'Intended Audience :: Developers',
		],
		include_dirs=[numpy.get_include()]
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

# Commande Bash : python setup_core.py build_ext --inplace
# if error: command 'clang++' failed: No such file or directory:
# CXX=g++ python setup_core.py build_ext --inplace
