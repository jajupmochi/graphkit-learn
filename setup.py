import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

with open('requirements_pypi.txt') as fp:
	install_requires = fp.read()

setuptools.setup(
	name="graphkit-learn",
	version="0.2b3",
	author="Linlin Jia",
	author_email="linlin.jia@insa-rouen.fr",
	description="A Python library for graph kernels, graph edit distances, and graph pre-images",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/jajupmochi/graphkit-learn",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Operating System :: OS Independent",
	],
	install_requires=install_requires,
	extras_require={
		'control>=0.8.2': ['matplotlib>=3.1.0'], # for generalized random walk kernels only.
		'control<0.8.2': ['matplotlib<3.1.0']
		},
)
