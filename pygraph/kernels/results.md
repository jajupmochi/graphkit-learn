# Results with minimal test RMSE for each kernel on dataset Asyclic
All kernels are tested on dataset Asyclic, which consists of 185 molecules (graphs). 

The criteria used for prediction are SVM for classification and kernel Ridge regression for regression.

For predition we randomly divide the data in train and test subset, where 90% of entire dataset is for training and rest for testing. 10 splits are performed. For each split, we first train on the train data, then evaluate the performance on the test set. We choose the optimal parameters for the test set and finally provide the corresponding performance. The final results correspond to the average of the performances on the test sets. 

## Summary

| Kernels       | RMSE(℃) | STD(℃) |    Parameter | k_time |
|---------------|:-------:|:------:|-------------:|-------:|
| Shortest path | 35.19   | 4.50   |            - | 14.58" |
| Marginalized  | 18.02   | 6.29   | p_quit = 0.1 |  4'19" |
| Path          | 14.00   | 6.94   |            - | 37.58" |
| WL subtree    | 7.55    | 2.33   |   height = 1 |  0.84" |
| Treelet       | 8.31    | 3.38   |            - |  0.50" |
| Path up to d  | 7.43    | 2.69   |    depth = 2 |  0.52" |

* RMSE stands for arithmetic mean of the root mean squared errors on all splits.
* STD stands for standard deviation of the root mean squared errors on all splits.
* Paremeter is the one with which the kenrel achieves the best results.
* k_time is the time spent on building the kernel matrix.
* The targets of training data are normalized before calculating *path kernel* and *treelet kernel*.

## Detailed results of each kernel
In each table below:
* The unit of the *RMSEs* and *stds* is *℃*, The unit of the *k_time* is *s*.
* k_time is the time spent on building the kernel matrix.

### shortest path kernel
```
  RMSE_test    std_test    RMSE_train    std_train    k_time
-----------  ----------  ------------  -----------  --------
     35.192     4.49577       28.3604      1.35718   14.5768
```

### Marginalized kernel
The table below shows the results of the marginalized under different termimation probability.
```
  p_quit    RMSE_test    std_test    RMSE_train    std_train    k_time
--------  -----------  ----------  ------------  -----------  --------
     0.1      18.0243     6.29247       12.1863      7.03899   258.77
     0.2      18.3376     5.85454       13.9554      7.54407   256.327
     0.3      18.496      5.73492       13.9391      7.95812   255.614
     0.4      19.4491     5.3713        16.2593      6.69358   254.897
     0.5      19.7857     5.55054       17.0181      6.84437   256.757
     0.6      20.1922     5.59122       17.6618      6.56718   256.557
     0.7      21.6614     6.02685       20.5882      5.74601   254.953
     0.8      22.996      6.08347       23.5943      3.80637   252.804
     0.9      24.4241     4.95119       25.8082      3.31207   256.738
```

### Path kernel
**The targets of training data are normalized before calculating the kernel.**
```
  RMSE_test    std_test    RMSE_train    std_train    k_time
-----------  ----------  ------------  -----------  --------
    14.0015     6.93602       3.76191     0.702594   37.5759
```

### Weisfeiler-Lehman subtree kernel
The table below shows the results of the WL subtree under different subtree heights.
```
  height    RMSE_test    std_test    RMSE_train    std_train    k_time
--------  -----------  ----------  ------------  -----------  --------
       0     15.6859      4.1392      17.6816       0.713183  0.360443
       1      7.55046     2.33179      6.27001      0.654734  0.837389
       2      9.72847     2.05767      4.45068      0.882129  1.25317
       3     11.2961      2.79994      2.27059      0.481516  1.79971
       4     12.8083      3.44694      1.07403      0.637823  2.35346
       5     14.0179      3.67504      0.700602     0.57264   2.78285
       6     14.9184      3.80535      0.691515     0.56462   3.20764
       7     15.6295      3.86539      0.691516     0.56462   3.71648
       8     16.2144      3.92876      0.691515     0.56462   3.99213
       9     16.7257      3.9931       0.691515     0.56462   4.26315
      10     17.1864      4.05672      0.691516     0.564621  5.00918
```

### Treelet kernel
 **The targets of training data are normalized before calculating the kernel.**
```
  RMSE_test    std_test    RMSE_train    std_train    k_time
-----------  ----------  ------------  -----------  --------
     8.3079     3.37838       2.90887       1.2679   0.500302
```

### Path kernel up to depth *d*
The table below shows the results of the path kernel up to different depth *d*.

The first table is the results using Tanimoto kernel, where **The targets of training data are normalized before calculating the kernel.**.
```
  depth    rmse_test    std_test    rmse_train    std_train     k_time
-------  -----------  ----------  ------------  -----------  ---------
      0      41.6202     6.453         43.6169      2.13212  0.0904737
      1      38.8446     6.44648       40.8329      3.44147  0.175414
      2      35.2915     4.7813        35.7461      1.61134  0.344896
      3      29.4845     3.90351       28.4646      3.00137  0.553939
      4      22.6693     6.28053       19.2517      3.42893  0.770649
      5      21.7956     5.5225        16.886       2.60519  1.01558
      6      20.6049     5.49983       13.1097      2.58431  1.33302
      7      20.3479     5.17631       12.0152      2.5928   1.60266
      8      19.8228     5.13769       10.7981      2.13082  1.81218
      9      19.8734     5.10369       10.7997      2.09549  2.21726
     10      19.8708     5.09217       10.7787      2.10002  2.41006
```

The second table is the results using MinMax kernel.
```
depth    rmse_test    std_test    rmse_train    std_train    k_time
-------  -----------  ----------  ------------  -----------  --------
      0     12.58        2.73235      12.1209      0.500467  0.377576
      1     12.6215      2.18866      10.2243      0.734261  0.456332
      2      7.42903     2.69395       2.71885     0.732922  0.585278
      3      9.02468     2.50808       1.54        1.13813   0.706556
      4     10.0811      3.6477        1.36029     1.42399   0.847957
      5     11.3005      4.44163       1.08518     1.06206   1.00086
      6     12.186       4.88816       1.06443     1.00191   1.19792
      7     12.7534      5.14529       1.19912     1.34031   1.4372
      8     13.0471      5.27184       1.35822     1.84315   1.68449
      9     13.1789      5.27707       1.36002     1.84834   1.96545
     10     13.2538      5.26425       1.36208     1.85426   2.24943
```
