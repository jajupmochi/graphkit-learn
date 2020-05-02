# graphkit-learn
[![Build Status](https://travis-ci.org/jajupmochi/graphkit-learn.svg?branch=master)](https://travis-ci.org/jajupmochi/graphkit-learn)
[![Build status](https://ci.appveyor.com/api/projects/status/bdxsolk0t1uji9rd?svg=true)](https://ci.appveyor.com/project/jajupmochi/graphkit-learn)
[![codecov](https://codecov.io/gh/jajupmochi/graphkit-learn/branch/master/graph/badge.svg)](https://codecov.io/gh/jajupmochi/graphkit-learn)
[![Documentation Status](https://readthedocs.org/projects/graphkit-learn/badge/?version=master)](https://graphkit-learn.readthedocs.io/en/master/?badge=master)
[![PyPI version](https://badge.fury.io/py/graphkit-learn.svg)](https://badge.fury.io/py/graphkit-learn)

A Python package for graph kernels, graph edit distances and graph pre-image problem.

## Requirements

* python>=3.5
* numpy>=1.16.2
* scipy>=1.1.0
* matplotlib>=3.0.0
* networkx>=2.2
* scikit-learn>=0.20.0
* tabulate>=0.8.2
* tqdm>=4.26.0
* control==0.8.0 (for generalized random walk kernels only)
* slycot==0.3.3 (for generalized random walk kernels only, which requires a fortran compiler, gfortran for example)

## How to use?

Simply clone this repository and voilà! Then check [`notebooks`](https://github.com/jajupmochi/graphkit-learn/tree/master/notebooks) directory for demos:
* [`notebooks`](https://github.com/jajupmochi/graphkit-learn/tree/master/notebooks) directory includes test codes of graph kernels based on linear patterns;
* [`notebooks/tests`](https://github.com/jajupmochi/graphkit-learn/tree/master/notebooks/tests) directory includes codes that test some libraries and functions;
* [`notebooks/utils`](https://github.com/jajupmochi/graphkit-learn/tree/master/notebooks/utils) directory includes some useful tools, such as a Gram matrix checker and a function to get properties of datasets;
* [`notebooks/else`](https://github.com/jajupmochi/graphkit-learn/tree/master/notebooks/else) directory includes other codes that we used for experiments.

## List of graph kernels

* Based on walks
  * The common walk kernel [1]
    * Exponential
    * Geometric
  * The marginalized kenrel
    * With tottering [2]
    * Without tottering [7]
  * The generalized random walk kernel [3]
    * Sylvester equation
    * Conjugate gradient
    * Fixed-point iterations
    * Spectral decomposition
* Based on paths
  * The shortest path kernel [4]
  * The structural shortest path kernel [5]
  * The path kernel up to length h [6]
    * The Tanimoto kernel
    * The MinMax kernel
* Non-linear kernels
  * The treelet kernel [10]
  * Weisfeiler-Lehman kernel [11]
    * Subtree

## Computation optimization methods

* Python’s `multiprocessing.Pool` module is applied to perform **parallelization** on the computations of all kernels as well as the model selection.
* **The Fast Computation of Shortest Path Kernel (FCSP) method** [8] is implemented in *the random walk kernel*, *the shortest path kernel*, as well as *the structural shortest path kernel* where FCSP is applied on both vertex and edge kernels.
* **The trie data structure** [9] is employed in *the path kernel up to length h* to store paths in graphs.

## Issues

* This library uses `multiprocessing.Pool.imap_unordered` function to do the parallelization, which may not be able to run correctly under Windows system. For now, Windows users may need to comment the parallel codes and uncomment the codes below them which run serially. We will consider adding a parameter to control serial or parallel computations as needed.

* Some modules (such as `Numpy`, `Scipy`, `sklearn`) apply [`OpenBLAS`](https://www.openblas.net/) to perform parallel computation by default, which causes conflicts with other parallelization modules such as `multiprossing.Pool`, highly increasing the computing time. By setting its thread to 1, `OpenBLAS` is forced to use a single thread/CPU, thus avoids the conflicts. For now, this procedure has to be done manually. Under Linux, type this command in terminal before running the code:
```
$ export OPENBLAS_NUM_THREADS=1
```
Or add `export OPENBLAS_NUM_THREADS=1` at the end of your `~/.bashrc` file, then run
```
$ source ~/.bashrc
```
to make this effective permanently.

## Results

Check this paper for detailed description of graph kernels and experimental results:

Linlin Jia, Benoit Gaüzère, and Paul Honeine. Graph Kernels Based on Linear Patterns: Theoretical and Experimental Comparisons. working paper or preprint, March 2019. URL https://hal-normandie-univ.archives-ouvertes.fr/hal-02053946.

A comparison of performances of graph kernels on benchmark datasets can be found [here](https://graphkit-learn.readthedocs.io/en/master/experiments.html).

## References
[1] Thomas Gärtner, Peter Flach, and Stefan Wrobel. On graph kernels: Hardness results and efficient alternatives. Learning Theory and Kernel Machines, pages 129–143, 2003.

[2] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between labeled graphs. In Proceedings of the 20th International Conference on Machine Learning, Washington, DC, United States, 2003.

[3] Vishwanathan, S.V.N., Schraudolph, N.N., Kondor, R., Borgwardt, K.M., 2010. Graph kernels. Journal of Machine Learning Research 11, 1201–1242.

[4] K. M. Borgwardt and H.-P. Kriegel. Shortest-path kernels on graphs. In Proceedings of the International Conference on Data Mining, pages 74-81, 2005.

[5] Liva Ralaivola, Sanjay J Swamidass, Hiroto Saigo, and Pierre Baldi. Graph kernels for chemical informatics. Neural networks, 18(8):1093–1110, 2005.

[6] Suard F, Rakotomamonjy A, Bensrhair A. Kernel on Bag of Paths For Measuring Similarity of Shapes. InESANN 2007 Apr 25 (pp. 355-360).

[7] Mahé, P., Ueda, N., Akutsu, T., Perret, J.L., Vert, J.P., 2004. Extensions of marginalized graph kernels, in: Proc. the twenty-first international conference on Machine learning, ACM. p. 70.

[8] Lifan Xu, Wei Wang, M Alvarez, John Cavazos, and Dongping Zhang. Parallelization of shortest path graph kernels on multi-core cpus and gpus. Proceedings of the Programmability Issues for Heterogeneous Multicores (MultiProg), Vienna, Austria, 2014.

[9] Edward Fredkin. Trie memory. Communications of the ACM, 3(9):490–499, 1960.

[10] Gaüzere, B., Brun, L., Villemin, D., 2012. Two new graphs kernels in chemoinformatics. Pattern Recognition Letters 33, 2038–2047.

[11] Shervashidze, N., Schweitzer, P., Leeuwen, E.J.v., Mehlhorn, K., Borgwardt, K.M., 2011. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research 12, 2539–2561.

## Authors

* [Linlin Jia](https://jajupmochi.github.io/), LITIS, INSA Rouen Normandie
* [Benoit Gaüzère](http://pagesperso.litislab.fr/~bgauzere/#contact_en), LITIS, INSA Rouen Normandie
* [Paul Honeine](http://honeine.fr/paul/Welcome.html), LITIS, Université de Rouen Normandie

## Citation

Still waiting...

## Acknowledgments

This research was supported by CSC (China Scholarship Council) and the French national research agency (ANR) under the grant APi (ANR-18-CE23-0014). The authors would like to thank the CRIANN (Le Centre Régional Informatique et d’Applications Numériques de Normandie) for providing computational resources.
