import sys
import setuptools
from datetime import datetime


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--build-gedlibpy',
		dest='build_gedlibpy',
		type=str,
		choices=['true', 'false'],
		default='true',
		help='Whether to build the Cython gedlibpy module. If `false`, then it '
		     'is possible that the module would not work when it is incompatible'
		     ' with the system (libraries).'
	)

	parser.add_argument(
		'--use-existing-gedlib',
		dest='use_existing_gedlib',
		type=str,
		choices=['true', 'false'],
		default='false',
		help='Whether to use an existing GEDLIB C++ library. This will avoid '
		     'downloading the library from GitHub and extracting it. If no library '
		     'is found, the installation will stop and an error will be raised. '
		     'Does not have effect when `--build-gedlibpy` is set to `false`. It may help '
		     'when you have problem accessing GitHub or it takes too long time to '
		     'extract the file (which is my case on CentOS (I hate this system!)). '
	)

	parser.add_argument(
		'--build-gedlib',
		dest='build_gedlib',
		type=str,
		choices=['true', 'false'],
		default='true',
		help='Whether to build GEDLIB C++. Does not have effect when '
		     '`--build-gedlibpy` is set to `false`.'
	)

	parser.add_argument(
		'--develop-mode',
		dest='develop_mode',
		type=str,
		choices=['true', 'false'],
		default='true',
		help='Whether in development mode. If true, the include files in the '
		     '`gedlibpy` module will be deleted after installation.  Does not '
		     'have effect when `--build-gedlibpy` is set to `false`. Notice, the '
		     'default value is `true`.'
	)

	args, unknown = parser.parse_known_args()
	return args


args = parse_args()

if args.build_gedlibpy == 'true':
	# Compile GEDLIBPY module:
	import subprocess

	cur_python = sys.executable
	subprocess.call([cur_python, '--version'])
	subprocess.call(['which', cur_python])
	gedlib_dir = 'gklearn/gedlib/'
	subprocess.call(
		[
			cur_python, 'setup.py',
			# '--use-existing-gedlib', args.use_existing_gedlib,
			# '--build-gedlib', args.build_gedlib,
			# '--develop-mode', args.develop_mode,
			'build_ext', '--inplace'
		], cwd=gedlib_dir
	)

# Install graphkit-learn:
with open("README.md", "r") as fh:
	long_description = fh.read()

with open('requirements_pypi.txt') as fp:
	install_requires = fp.read()

version = '0.2.1.post' + datetime.now().strftime('%Y%m%d%H%M%S')
setuptools.setup(
	name="graphkit-learn",
	version=version,
	author="Linlin Jia",
	author_email="linlin.jia@unibe.ch",
	description="A Python library for graph kernels, graph edit distances, and graph pre-images",
	long_description=long_description,
	long_description_content_type="text/markdown",
	project_urls={
		'Documentation': 'https://graphkit-learn.readthedocs.io',
		'Source': 'https://github.com/jajupmochi/graphkit-learn',
		'Tracker': 'https://github.com/jajupmochi/graphkit-learn/issues',
	},
	url="https://github.com/jajupmochi/graphkit-learn",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Operating System :: OS Independent",
		'Intended Audience :: Science/Research',
		'Intended Audience :: Developers',
	],
	install_requires=install_requires,
)
