import setuptools
from datetime import datetime

# Compile GEDLIBPY module:
import subprocess

gedlib_dir = 'gklearn/gedlib/'
subprocess.call(
	['python', 'setup.py', 'build_ext', '--inplace'], cwd=gedlib_dir
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
