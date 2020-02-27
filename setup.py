import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphkit-learn",
    version="0.1b2",
    author="Linlin Jia",
    author_email="linlin.jia@insa-rouen.fr",
    description="A Python library for graph kernels based on linear patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jajupmochi/graphkit-learn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
