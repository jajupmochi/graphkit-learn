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

	print()
	# Delete previous GEDLIB installation if exists:
	path_gedlib = os.path.join(os.getcwd(), 'include/gedlib-master/')
	if os.path.exists(path_gedlib):
		print('Deleting previous GEDLIB installation...')
		shutil.rmtree(path_gedlib)
		print('Done!')
		print()
	else:
		pass
		print('No previous installation of GEDLIB.')

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

	# Delete the .zip file:
	os.remove(filename)
	print('The .zip file of GEDLIB is deleted.')

	print('Done!')
	print()


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
	print()
	print('Installing GEDLIB...')
	import subprocess
	cur_python = sys.executable
	subprocess.call([cur_python, '--version'])
	subprocess.call(['which', cur_python])
	gedlib_dir = 'include/gedlib-master/'
	subprocess.call(
		[cur_python, 'install.py'], cwd=gedlib_dir
	)
	print()


def copy_gedlib_includes_and_libs():
	def copy_paste_with_logs(src_path, dst_path):
		"""
		Authors
		-------
		Linlin Jia, ChatGPT 3.5 (2023.04.27)
		"""
		cur_dir = os.path.dirname(os.path.abspath(__file__))
		src_path = os.path.join(cur_dir, src_path)
		dst_path = os.path.join(cur_dir, dst_path)

		# if the source is a file, copy it and print the copied file name
		if os.path.isfile(src_path):
			shutil.copy(src_path, dst_path)
			print(f"Copied file {src_path} to {dst_path}.")

		# if the source is a directory, copy it and print the copied file names
		if os.path.isdir(src_path):
			shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
			print(f"Copied directory {src_path} to {dst_path}.")


	print()
	print(
		'Copying and pasting (generated) includes and libs files of GEDLIB into '
		'proper locations to build gedlibpy...'
	)
	# Copy-paste includes:
	print('Copying includes...')
	copy_paste_with_logs(
		'include/gedlib-master/ext/boost.1.69.0/', 'include/boost.1.69.0/'
	)
	copy_paste_with_logs(
		'include/gedlib-master/ext/eigen.3.3.4/Eigen/',
		'include/eigen.3.3.4/Eigen/'
	)
	copy_paste_with_logs(
		'include/gedlib-master/ext/nomad.3.8.1/src/', 'include/nomad.3.8.1/src/'
	)
	copy_paste_with_logs(
		'include/gedlib-master/ext/nomad.3.8.1/ext/sgtelib/src/',
		'include/nomad.3.8.1/ext/sgtelib/src/'
	)
	copy_paste_with_logs(
		'include/gedlib-master/ext/lsape.5/include/', 'include/lsape.5/include/'
	)
	copy_paste_with_logs(
		'include/gedlib-master/ext/libsvm.3.22/', 'include/libsvm.3.22/'
	)
	copy_paste_with_logs(
		'include/gedlib-master/ext/fann.2.2.0/include/',
		'include/fann.2.2.0/include/'
	)
	# Copy-paste libs:
	print('Copying includes...')
	copy_paste_with_logs(
		'include/gedlib-master/ext/nomad.3.8.1/lib/', 'lib/nomad.3.8.1/'
	)
	copy_paste_with_logs(
		'include/gedlib-master/ext/libsvm.3.22/', 'lib/libsvm.3.22/'
	)
	copy_paste_with_logs(
		'include/gedlib-master/ext/fann.2.2.0/lib/', 'lib/fann.2.2.0/'
	)

	print('done!')
	print()


def clean_previous_build():
	# clean previous build
	print()
	print('Cleaning previous build...')
	for name in os.listdir():
		if (name.startswith('gedlibpy') and not (
				name.endswith('.pyx') or name.endswith('.pxd'))):
			os.remove(name)
		if name == 'build':
			shutil.rmtree(name)
	print('Done!')
	print()


def get_extensions():
	extensions = [
		Extension(
			"gedlibpy",
			# sources=["gedlibpy.pyx", "src/GedLibBind.cpp"],
			sources=["gedlibpy.pyx"],
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
			# library_dirs=["."],
			# libraries=["gxlgedlib"],
			language="c++",
			extra_compile_args=["-std=c++11"],
			extra_link_args=["-std=c++11"]
		)
	]
	return extensions


def remove_includes():
	pass


if __name__ == '__main__':

	# Download GEDLIB and unpack it:
	get_gedlib()
	# Install GEDLIB:
	install_gedlib()
	# Copy-Paste includes and libs of GEDLIB:
	copy_gedlib_includes_and_libs()
	# clean previous build:
	clean_previous_build()

	print()
	print('Start building...')
	# Build gedlibpy:
	extensions = ()
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
	print()
	print('The following files are generated:')
	for name in os.listdir():
		if (name.startswith('gedlibpy') and not (
				name.endswith('.pyx') or name.endswith('.pxd'))):
			print(name)

	print()
	print('Build completed!')
	print()

	# Remove GEDLIB include files:
	remove_includes()

# Commande Bash : python setup.py build_ext --inplace
