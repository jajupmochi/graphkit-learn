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
All kernels are tested on dataset Asyclic, which consists of 185 molecules (graphs). 

The criteria used for prediction are SVM for classification and kernel Ridge regression for regression.

For predition we randomly divide the data in train and test subset, where 90% of entire dataset is for training and rest for testing. 10 splits are performed. For each split, we first train on the train data, then evaluate the performance on the test set. We choose the optimal parameters for the test set and finally provide the corresponding performance. The final results correspond to the average of the performances on the test sets. 

| Kernels       | RMSE(℃)  | STD(℃)  | Parameter    | k_time |
|---------------|:---------:|:--------:|-------------:|-------:|
| Shortest path | 35.19     | 4.50     | -            | 14.58" |
| Marginalized  | 18.02     | 6.29     | p_quit = 0.1 | 4'19"  |
| Path          | 14.00     | 6.93     | -            | 36.21" |
| WL subtree    | 7.55      | 2.33     | height = 1   | 0.84"  |
| Treelet       | 8.31      | 3.38     | -            | 49.58" |

* RMSE stands for arithmetic mean of the root mean squared errors on all splits.
* STD stands for standard deviation of the root mean squared errors on all splits.
* Paremeter is the one with which the kenrel achieves the best results.
* k_time is the time spent on building the kernel matrix.
* The targets of training data are normalized before calculating *path kernel* and *treelet kernel*.
* See detail results in [results.md](pygraph/kernels/results.md).

## References
[1] K. M. Borgwardt and H.-P. Kriegel. Shortest-path kernels on graphs. In Proceedings of the International Conference on Data Mining, pages 74-81, 2005.

[2] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between labeled graphs. In Proceedings of the 20th International Conference on Machine Learning, Washington, DC, United States, 2003.

[3] Suard F, Rakotomamonjy A, Bensrhair A. Kernel on Bag of Paths For Measuring Similarity of Shapes. InESANN 2007 Apr 25 (pp. 355-360).

[4] N. Shervashidze, P. Schweitzer, E. J. van Leeuwen, K. Mehlhorn, and K. M. Borgwardt. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12:2539-2561, 2011.

[5] Gaüzère B, Brun L, Villemin D. Two new graphs kernels in chemoinformatics. Pattern Recognition Letters. 2012 Nov 1;33(15):2038-47.

## Updates
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
