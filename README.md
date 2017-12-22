# py-graph
a python package for graph kernels.

## requirements

* numpy - 1.13.3
* scipy - 1.0.0
* matplotlib - 2.1.0
* networkx - 2.0
* sklearn - 0.19.1
* tabulate - 0.8.2

## results with minimal test RMSE for each kernel on dataset Asyclic
-- All the kernels are tested on dataset Asyclic, which consists of 185 molecules (graphs). 
-- The criteria used for prediction are SVM for classification and kernel Ridge regression for regression.
-- For predition we randomly divide the data in train and test subset, where 90% of entire dataset is for training and rest for testing. 10 splits are performed. For each split, we first train on the train data, then evaluate the performance on the test set. We choose the optimal parameters for the test set and finally provide the corresponding performance. The final results correspond to the average of the performances on the test sets. 

| Kernels       | RMSE(℃)  | std(℃)  | parameter    | k_time |
|---------------|:---------:|:--------:|-------------:|-------:|
| shortest path | 36.40     | 5.35     | -            | -      |
| marginalized  | 17.90     | 6.59     | p_quit = 0.1 | -      |
| path          | 14.27     | 6.37     | -            | -      |
| WL subtree    | 9.00      | 6.37     | height = 1   | 0.85"  |

**In each line, paremeter is the one with which the kenrel achieves the best results.
In each line, k_time is the time spent on building the kernel matrix.
See detail results in [results.md](pygraph/kernels/results.md).**

## updates
### 2017.12.22
* ADD calculation of the time spend to acquire kernel matrices for each kernel. - linlin
* MOD floydTransformation function, calculate shortest paths taking into consideration user-defined edge weight. - linlin
* MOD implementation of nodes and edges attributes genericity for all kernels. - linlin
* ADD detailed results file results.md. - linlin
### 2017.12.21
* MOD Weisfeiler-Lehman subtree kernel and the test code. - linlin
### 2017.12.20
* ADD Weisfeiler-Lehman subtree kernel and its result on dataset Asyclic. - linlin
### 2017.12.07
* ADD mean average path kernel and its result on dataset Asyclic. - linlin
* ADD delta kernel. - linlin
* MOD reconstruction the code of marginalized kernel. - linlin
### 2017.12.05
* ADD marginalized kernel and its result. - linlin
* ADD list required python packages in file README.md. - linlin
### 2017.11.24
* ADD shortest path kernel and its result. - linlin
