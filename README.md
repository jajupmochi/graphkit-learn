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

~~For predition we randomly divide the data in train and test subset, where 90% of entire dataset is for training and rest for testing. 10 splits are performed. For each split, we first train on the train data, then evaluate the performance on the test set. We choose the optimal parameters for the test set and finally provide the corresponding performance. The final results correspond to the average of the performances on the test sets.~~

| Kernels          | train_perf | valid_perf |  test_perf |                                            Parameters | gram_matrix_time |
|------------------|-----------:|-----------:|-----------:|------------------------------------------------------:|-----------------:|
| Shortest path    | 28.65±0.59 | 36.09±0.97 | 36.45±6.63 |                                   'alpha': '3.55e+01' |           12.67" |
| Marginalized     | 12.42±0.28 | 18.60±2.02 | 16.51±5.12 |                    'p_quit': 0.3, 'alpha': '3.16e-06' |          430.42" |
| Path             | 11.19±0.73 | 23.66±1.74 | 25.04±9.60 |                                   'alpha': '2.57e-03' |           21.84" |
| WL subtree       |  6.00±0.27 |  7.59±0.71 |  7.92±2.92 |                    'height': 1.0, 'alpha': '1.26e-01' |            0.84" |
| WL shortest path | 28.32±0.63 | 35.99±0.98 | 37.92±5.60 |                    'height': 2.0, 'alpha': '1.00e+02' |           39.79" |
| WL edge          | 30.10±0.57 | 35.13±0.78 | 37.70±6.92 |                    'height': 4.0, 'alpha': '3.98e+01' |            4.35" |
| Treelet          |  7.38±0.37 | 14.21±0.80 | 15.26±3.65 |                                   'alpha': '1.58e+00' |            0.49" |
| Path up to d     |  5.48±0.23 | 10.00±0.83 | 10.73±5.67 | 'depth': 2.0, 'k_func': 'MinMax', 'alpha': '7.94e-02' |            0.57" |
| Tree pattern     |            |            |            |                                                       |                  |
| Cyclic pattern   |  0.62±0.02 |  0.62±0.02 |  0.57±0.17 |                 'cycle_bound': 125.0, 'C': '1.78e-01' |            0.33" |
* RMSE stands for arithmetic mean of the root mean squared errors on all splits.
* STD stands for standard deviation of the root mean squared errors on all splits.
* Paremeters are the ones with which the kenrel achieves the best results.
* gram_matrix_time is the time spent on building the gram matrix.
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

## Updates
### 2018.01.24
* ADD *path kernel up to depth d* and its result on dataset Asyclic.
* MOD treelet kernel, retrieve canonkeys of all graphs before calculate kernels, wildly speed it up.
### 2018.01.17
* ADD comments to code of treelet kernel. - linlin
### 2018.01.16
* ADD *treelet kernel* and its result on dataset Asyclic. - linlin
* MOD the way to calculate WL subtree kernel, correct its results. - linlin
* ADD *kernel_train_test* and *split_train_test* to wrap training and testing process. - linlin
* MOD readme.md file, add detailed results of each kernel. - linlin
### 2017.12.22
* ADD calculation of the time spend to acquire kernel matrices for each kernel. - linlin
* MOD floydTransformation function, calculate shortest paths taking into consideration user-defined edge weight. - linlin
* MOD implementation of nodes and edges attributes genericity for all kernels. - linlin
* ADD detailed results file results.md. - linlin
### 2017.12.21
* MOD Weisfeiler-Lehman subtree kernel and the test code. - linlin
### 2017.12.20
* ADD *Weisfeiler-Lehman subtree kernel* and its result on dataset Asyclic. - linlin
### 2017.12.07
* ADD *mean average path kernel* and its result on dataset Asyclic. - linlin
* ADD delta kernel. - linlin
* MOD reconstruction the code of marginalized kernel. - linlin
### 2017.12.05
* ADD *marginalized kernel* and its result. - linlin
* ADD list required python packages in file README.md. - linlin
### 2017.11.24
* ADD *shortest path kernel* and its result. - linlin
