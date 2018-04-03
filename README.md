# py-graph
A python package for graph kernels.

## Requirements

* numpy - 1.13.3
* scipy - 1.0.0
* matplotlib - 2.1.0
* networkx - 2.0
* sklearn - 0.19.1
* tabulate - 0.8.2

## Results with minimal test RMSE for each kernel on dataset Asyclic

All kernels expect for Cyclic pattern kernel are tested on dataset Asyclic, which consists of 185 molecules (graphs). (Cyclic pattern kernel is tested on dataset MAO and PAH.)

The criteria used for prediction are SVM for classification and kernel Ridge regression for regression.

For prediction we randomly divide the data in train and test subset, where 90\% of entire dataset is for training and rest for testing. 30 splits are performed. For each split, we first train on the train data, then evaluate the performance on the test set. We choose the optimal parameters for the test set and finally provide the corresponding performance. The final results correspond to the average of the performances on the test sets.

| Kernels                   | train_perf | valid_perf | test_perf  | Parameters                                       | gram_matrix_time          |
|---------------------------|------------|------------|------------|--------------------------------------------------|---------------------------|
| Shortest path             | 28.77±0.60 | 38.31±0.92 | 39.40±6.32 | 'alpha': '1.00'                                  | 13.54"                    |
| Marginalized              | 12.95±0.37 | 19.02±1.73 | 18.24±5.00 | 'p_quit': 0.2, 'alpha': '1.00e-04'               | 437.04"/447.44"±5.32"     |
| Extension of Marginalized | 20.65±0.44 | 26.06±1.83 | 26.84±4.81 | 'p_quit': 0.1, 'alpha': '5.62e-04'               | 6388.50"/6266.67"±149.16" |
| Path                      | 8.71±0.63  | 19.28±1.75 | 17.42±6.57 | 'alpha': '2.82e-02'                              | 21.94"                    |
| WL subtree                | 13.90±0.35 | 18.47±1.36 | 18.08±4.70 | 'height': 1.0, 'alpha': '1.50e-03'               | 0.79"/1.32"±0.76"         |
| WL shortest path          | 28.74±0.60 | 38.20±0.62 | 39.02±6.09 | 'height': 10.0, 'alpha': '1.00'                  | 146.83"/80.63"±45.04"     |
| WL edge                   | 30.21±0.64 | 36.53±1.02 | 38.42±6.42 | 'height': 5.0, 'alpha': '6.31e-01'               | 5.24"/5.15"±2.83"         |
| Treelet                   | 7.33±0.64  | 13.86±0.80 | 15.38±3.56 | 'alpha': '1.12e+01'                              | 0.48"                     |
| Path up to d              | 5.76±0.27  | 9.89±0.87  | 10.21±4.16 | 'depth': 2.0, 'k_func': 'MinMax', 'alpha': '0.1' | 0.56"/1.16"±0.75"         |
| Cyclic pattern            |            |            |            |                                                  |                           |
| Walk up to n              | 20.88±0.74 | 23.34±1.11 | 24.46±6.57 | 'n': 2.0, 'alpha': '1.00e-03'                    | 0.56"/331.70"±753.44"     |

In table above,last column is the time consumed to calculate the gram matrix. Note for
kernels which need to tune hyper-parameters that are required to calculate gram
matrices, average time consumption and its confidence are obtained over the
hyper-parameters grids, which are shown after "/". The time shown before "/"
is the one spent on building the gram matrix corresponding to the best test
performance.

* See detail results in [results.md](pygraph/kernels/results.md).

## References
[1] K. M. Borgwardt and H.-P. Kriegel. Shortest-path kernels on graphs. In Proceedings of the International Conference on Data Mining, pages 74-81, 2005.

[2] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between labeled graphs. In Proceedings of the 20th International Conference on Machine Learning, Washington, DC, United States, 2003.

[3] Suard F, Rakotomamonjy A, Bensrhair A. Kernel on Bag of Paths For Measuring Similarity of Shapes. InESANN 2007 Apr 25 (pp. 355-360).

[4] N. Shervashidze, P. Schweitzer, E. J. van Leeuwen, K. Mehlhorn, and K. M. Borgwardt. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12:2539-2561, 2011.

[5] Gaüzère B, Brun L, Villemin D. Two new graphs kernels in chemoinformatics. Pattern Recognition Letters. 2012 Nov 1;33(15):2038-47.

[6] Liva Ralaivola, Sanjay J Swamidass, Hiroto Saigo, and Pierre Baldi. Graph kernels for chemical informatics. Neural networks, 18(8):1093–1110, 2005.

[7] Pierre Mahé and Jean-Philippe Vert. Graph kernels based on tree patterns for molecules. Machine learning, 75(1):3–35, 2009.

[8] Tamás Horváth, Thomas Gärtner, and Stefan Wrobel. Cyclic pattern kernels for predictive graph mining. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 158–167. ACM, 2004.

[9] Thomas Gärtner, Peter Flach, and Stefan Wrobel. On graph kernels: Hardness results and efficient alternatives. Learning Theory and Kernel Machines, pages 129–143, 2003.

## Updates
### 2018.02.28
* ADD *walk kernel up to n* and its result on dataset Asyclic.
* MOD training process, use nested cross validation for model selection. Recalculate performance of all kernels.
### 2018.02.08
* ADD *tree pattern kernel* and its result on dataset Asyclic.
* ADD *cyclic pattern kernel* and its result on classification datasets.
### 2018.01.24
* ADD *path kernel up to depth d* and its result on dataset Asyclic.
* MOD treelet kernel, retrieve canonkeys of all graphs before calculate kernels, wildly speed it up.
### 2018.01.17
* ADD comments to code of treelet kernel.
### 2018.01.16
* ADD *treelet kernel* and its result on dataset Asyclic.
* MOD the way to calculate WL subtree kernel, correct its results.
* ADD *kernel_train_test* and *split_train_test* to wrap training and testing process.
* MOD readme.md file, add detailed results of each kernel. - linlin
### 2017.12.22
* ADD calculation of the time spend to acquire kernel matrices for each kernel.
* MOD floydTransformation function, calculate shortest paths taking into consideration user-defined edge weight.
* MOD implementation of nodes and edges attributes genericity for all kernels.
* ADD detailed results file results.md.
### 2017.12.21
* MOD Weisfeiler-Lehman subtree kernel and the test code.
### 2017.12.20
* ADD *Weisfeiler-Lehman subtree kernel* and its result on dataset Asyclic.
### 2017.12.07
* ADD *mean average path kernel* and its result on dataset Asyclic.
* ADD delta kernel. - linlin
* MOD reconstruction the code of marginalized kernel.
### 2017.12.05
* ADD *marginalized kernel* and its result. - linlin
* ADD list required python packages in file README.md.
### 2017.11.24
* ADD *shortest path kernel* and its result.
