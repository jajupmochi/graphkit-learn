# py-graph
a python package for graph kernels.

## requirements

* numpy - 1.13.3
* scipy - 1.0.0
* matplotlib - 2.1.0
* networkx - 2.0
* sklearn - 0.19.1
* tabulate - 0.8.2

## results with minimal RMSE for each kernel on dataset Asyclic
| Kernels       | RMSE(℃)  | std(℃)  | parameter    |
|---------------|:---------:|:--------:|-------------:|
| shortest path | 36.400524 | 5.352940 | -            |
| marginalized  | 17.8991   | 6.59104  | p_quit = 0.1 |
| path          | 14.270816 | 6.366698 | -            |
| WL subtree    | 9.01403   | 6.35786  | height = 1   |

## updates
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
