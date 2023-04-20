from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize

import os
import shutil
import numpy

# Download GEDLIB and unpack it.
import importlib


def install(package):
	try:
		importlib.import_module(package)
	except ImportError:
		import subprocess
		subprocess.call(['pip', 'install', package])


install('tqdm')

import urllib.request
from tqdm import tqdm
import zipfile

print('Downloading and unpacking GEDLIB from GitHub...')

url = 'https://github.com/jajupmochi/gedlib/archive/refs/heads/master.zip'
filename = 'gedlib-master.zip'

with tqdm(
		unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename
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
			"src", "include", "include/lsape", "include/Eigen",
			"include/nomad", "include/sgtelib", "include/libsvm.3.22",
			"include/fann", "include/boost_1_69_0"
		],
		library_dirs=["lib/fann", "lib/gedlib", "lib/libsvm.3.22", "lib/nomad"],
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
