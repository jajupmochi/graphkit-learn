from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize

import os
import sys
import shutil
import numpy

import importlib


def install(package):
	try:
		importlib.import_module(package)
	except ImportError:
		import subprocess
		cur_python = sys.executable
		subprocess.call([cur_python, '-m', 'pip', 'install', package])


def get_gedlib():
	install('tqdm')

	import urllib.request
	from tqdm import tqdm
	import zipfile

	print('Downloading and unpacking GEDLIB from GitHub...')

	url = 'https://github.com/jajupmochi/gedlib/archive/refs/heads/master.zip'
	filename = 'gedlib-master.zip'

	with tqdm(
			unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
			desc=filename
	) as t:
		urllib.request.urlretrieve(
			url, filename=filename,
			reporthook=lambda blocknum, blocksize, totalsize: t.update(
				blocknum * blocksize - t.n
			)
		)

	with zipfile.ZipFile(filename, 'r') as zip_ref:
		for member in tqdm(zip_ref.infolist(), desc='Extracting'):
			try:
				zip_ref.extract(member, path='include/')
			except zipfile.error as e:
				pass
	# zip_ref.extractall('include/')
	print('Done!')


def install_gedlib():
	"""
	Notes
	-----
	The followings are done during this installation:
		- Build files in `ext/fann.2.2.0/lib/`;
		- Clean obj files in `ext/nomad.3.8.1/ext/sgtelib/src/`;
		- Build files in `ext/nomad.3.8.1/ext/sgtelib/lib/`;
		- Generate executable files in `ext/nomad.3.8.1/ext/sgtelib/bin/`;
		- Build files in `ext/nomad.3.8.1/lib/`;
		- Generate `ext/libsvm.3.22/libsvm.so`;
	"""
	import subprocess
	cur_python = sys.executable
	subprocess.call([cur_python, '--version'])
	subprocess.call(['which', cur_python])
	gedlib_dir = 'include/gedlib-master/'
	subprocess.call(
		[cur_python, 'install.py'], cwd=gedlib_dir
	)


# Download GEDLIB and unpack it.
get_gedlib()
# Install GEDLIB:
install_gedlib()

# clean previous build
print('Cleaning previous build...')
for name in os.listdir():
	if (name.startswith('gedlibpy') and not (
			name.endswith('.pyx') or name.endswith('.pxd'))):
		os.remove(name)
	if name == 'build':
		shutil.rmtree(name)
print('Done!')

print('Start building...')
extensions = [
	Extension(
		"gedlibpy",
		# sources=["gedlibpy.pyx", "src/GedLibBind.cpp"],
		sources=["gedlibpy.pyx"],
		include_dirs=[
			"src", "include", "include/lsape.5/include",
			"include/eigen.3.3.4/Eigen",
			"include/nomad.3.8.1/src", "include/nomad.3.8.1/ext/sgtelib/src",
			"include/libsvm.3.22",
			"include/fann.2.2.0/include", "include/boost.1.69.0"
		],
		library_dirs=[
			"lib/fann.2.2.0", "lib/libsvm.3.22", # "lib/gedlib",
			"lib/nomad.3.8.1"
		],
		libraries=["doublefann", "sgtelib", "svm", "nomad"],
		# library_dirs=["."],
		# libraries=["gxlgedlib"],
		language="c++",
		extra_compile_args=["-std=c++11"],
		extra_link_args=["-std=c++11"]
	)
]

with open("README.rst", "r") as fh:
	long_description = fh.read()

setup(
	ext_modules=cythonize(
		extensions,
		compiler_directives={'language_level': '3'}
	),
	name="gedlibpy",
	author="Lambert Natacha and Linlin Jia",
	author_email="linlin.jia@unibe.ch",
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
print('The following files are generated:')
for name in os.listdir():
	if (name.startswith('gedlibpy') and not (
			name.endswith('.pyx') or name.endswith('.pxd'))):
		print(name)

print('Build completed!')

# Commande Bash : python setup.py build_ext --inplace
