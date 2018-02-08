# Results with minimal test RMSE for each kernel on dataset Asyclic
All kernels expect for Cyclic pattern kernel are tested on dataset Asyclic, which consists of 185 molecules (graphs). (Cyclic pattern kernel is tested on dataset MAO and PAH.)

The criteria used for prediction are SVM for classification and kernel Ridge regression for regression.

For predition we randomly divide the data in train and test subset, where 90% of entire dataset is for training and rest for testing. 10 splits are performed. For each split, we first train on the train data, then evaluate the performance on the test set. We choose the optimal parameters for the test set and finally provide the corresponding performance. The final results correspond to the average of the performances on the test sets. 

All the results were run under Python 3.5.2, in a machine of 64 bit with one Intel(R) Core(TM) i7-7920HQ CPU @ 3.10GHz, Memory of 32GB, and Ubuntu 16.04.3 LTS OS.

## Summary

| Kernels          | RMSE(℃) | STD(℃) |         Parameter | k_time |
|------------------|:-------:|:------:|------------------:|-------:|
| Shortest path    | 35.19   | 4.50   |                 - | 14.58" |
| Marginalized     | 18.02   | 6.29   |      p_quit = 0.1 |  4'19" |
| Path             | 14.00   | 6.94   |                 - | 37.58" |
| WL subtree       | 7.55    | 2.33   |        height = 1 |  0.84" |
| WL shortest path | 35.16   | 4.50   |        height = 2 | 40.24" |
| WL edge          | 33.41   | 4.73   |        height = 5 |  5.66" |
| Treelet          | 8.31    | 3.38   |                 - |  0.50" |
| Path up to d     | 7.43    | 2.69   |         depth = 2 |  0.52" |
| Tree pattern     | 7.27    | 2.21   |  lamda = 1, h = 2 | 37.24" |
| Cyclic pattern   | 0.9     | 0.11   | cycle bound = 100 |  0.31" |

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

### Weisfeiler-Lehman shortest path kernel
The table below shows the results of the WL subtree under different subtree heights.
```
  height    rmse_test    std_test    rmse_train    std_train    k_time
--------  -----------  ----------  ------------  -----------  --------
       0      35.192      4.49577       28.3604      1.35718   13.5041
       1      35.1808     4.50045       27.9335      1.44836   26.8292
       2      35.1632     4.50205       28.1113      1.50891   40.2356
       3      35.1946     4.49801       28.3903      1.36571   54.6704
       4      35.1753     4.50111       27.9746      1.46222   67.1522
       5      35.1997     4.5071        28.0184      1.45564   80.0881
       6      35.1645     4.49849       28.3731      1.60057   92.1925
       7      35.1771     4.5009        27.9604      1.45742  105.812
       8      35.1968     4.50526       28.1991      1.5149   119.022
       9      35.1956     4.50197       28.2665      1.30769  131.228
      10      35.1676     4.49723       28.4163      1.61596  144.964
```

### Weisfeiler-Lehman edge kernel
The table below shows the results of the WL subtree under different subtree heights.
```
  height    rmse_test    std_test    rmse_train    std_train     k_time
--------  -----------  ----------  ------------  -----------  ---------
       0      33.4077     4.73272       29.9975     0.90234    0.853002
       1      33.4235     4.72131       30.1603     1.09423    1.71751
       2      33.433      4.72441       29.9286     0.787941   2.66032
       3      33.4073     4.73243       30.0114     0.909674   3.47763
       4      33.4256     4.72166       30.1842     1.1089     4.54367
       5      33.4067     4.72641       30.0411     1.01845    5.66178
       6      33.419      4.73075       29.9056     0.782179   6.14803
       7      33.4248     4.72155       30.1759     1.10382    7.60354
       8      33.4122     4.71554       30.1365     1.07485    7.97222
       9      33.4071     4.73193       30.0329     0.921065   9.07084
      10      33.4165     4.73169       29.9242     0.790843  10.0254
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

The first table is the results using *Tanimoto kernel*, where **The targets of training data are normalized before calculating the kernel.**.
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

The second table is the results using *MinMax kernel*.
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


### Tree pattern kernel
Until N kernel when h = 2:
```
       lmda    rmse_test    std_test    rmse_train    std_train    k_time
-----------  -----------  ----------  ------------  -----------  --------
     1e-10       7.46524     1.71862       5.99486     0.356634   38.1447
     1e-09       7.37326     1.77195       5.96155     0.374395   37.4921
     1e-08       7.35105     1.78349       5.96481     0.378047   37.9971
     1e-07       7.35213     1.77903       5.96728     0.382251   38.3182
     1e-06       7.3524      1.77992       5.9696      0.3863     39.6428
     1e-05       7.34958     1.78141       5.97114     0.39017    37.3711
     0.0001      7.3513      1.78136       5.94251     0.331843   37.3967
     0.001       7.35822     1.78119       5.9326      0.32534    36.7357
     0.01        7.37552     1.79037       5.94089     0.34763    36.8864
     0.1         7.32951     1.91346       6.42634     1.29405    36.8382
     1           7.27134     2.20774       6.62425     1.2242     37.2425
    10           7.49787     2.36815       6.81697     1.50182    37.8286
   100           7.42887     2.64789       6.68766     1.34809    36.3701
  1000           7.24914     2.65554       6.81906     1.41008    36.1695
 10000           7.08183     2.6248        6.93431     1.38441    37.5723
100000           8.021       3.43694       8.69813     0.909839   37.8158
     1e+06       8.49625     3.6332        9.59333     0.96626    38.4688
     1e+07      10.9067      3.17593      11.5642      2.07792    36.9926
     1e+08      61.1524     10.4355       65.3527     13.9538     37.1321
     1e+09      99.943      13.6994       98.8848      5.27014    36.7443
     1e+10     100.083      13.8503       97.9168      3.22768    37.096
```

### Cyclic pattern kernel
**This kernel is not tested on dataset Acyclic**

Results on dataset MAO:
```
cycle_bound    accur_test    std_test    accur_train    std_train    k_time
-------------  ------------  ----------  -------------  -----------  --------
            0      0.642857    0.146385       0.54918     0.0167983  0.187052
           50      0.871429    0.1            0.698361    0.116889   0.300629
          100      0.9         0.111575       0.732787    0.0826366  0.309837
          150      0.9         0.111575       0.732787    0.0826366  0.31808
          200      0.9         0.111575       0.732787    0.0826366  0.317575
```

Results on dataset PAH:
```
  cycle_bound    accur_test    std_test    accur_train    std_train    k_time
-------------  ------------  ----------  -------------  -----------  --------
            0          0.61    0.113578       0.629762    0.0135212  0.521801
           10          0.61    0.113578       0.629762    0.0135212  0.52589
           20          0.61    0.113578       0.629762    0.0135212  0.548528
           30          0.64    0.111355       0.633333    0.0157935  0.535311
           40          0.64    0.111355       0.633333    0.0157935  0.61764
           50          0.67    0.09           0.658333    0.0345238  0.733868
           60          0.68    0.107703       0.671429    0.0365769  0.871147
           70          0.67    0.100499       0.666667    0.0380208  1.12625
           80          0.78    0.107703       0.709524    0.0588534  1.19828
           90          0.78    0.107703       0.709524    0.0588534  1.21182
```
