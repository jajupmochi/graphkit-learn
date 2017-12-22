# results with minimal test RMSE for each kernel on dataset Asyclic
-- All the kernels are tested on dataset Asyclic, which consists of 185 molecules (graphs). 
-- The criteria used for prediction are SVM for classification and kernel Ridge regression for regression.
-- For predition we randomly divide the data in train and test subset, where 90% of entire dataset is for training and rest for testing. 10 splits are performed. For each split, we first train on the train data, then evaluate the performance on the test set. We choose the optimal parameters for the test set and finally provide the corresponding performance. The final results correspond to the average of the performances on the test sets. 

## summary

| Kernels       | RMSE(℃)  | std(℃)  | parameter    | k_time |
|---------------|:---------:|:--------:|-------------:|-------:|
| shortest path | 36.40     | 5.35     | -            | -      |
| marginalized  | 17.90     | 6.59     | p_quit = 0.1 | -      |
| path          | 14.27     | 6.37     | -            | -      |
| WL subtree    | 9.00      | 6.37     | height = 1   | 0.85"  |

**In each line, paremeter is the one with which the kenrel achieves the best results.
In each line, k_time is the time spent on building the kernel matrix.**

## detailed results of WL subtree kernel.
The table below shows the results of the WL subtree under different subtree heights.
```
  height    RMSE_test    std_test    RMSE_train    std_train    k_time
--------  -----------  ----------  ------------  -----------  --------
       0     36.2108      7.33179       141.419     1.08284   0.392911
       1      9.00098     6.37145       140.065     0.877976  0.812077
       2     19.8113      4.04911       140.075     0.928821  1.36955
       3     25.0455      4.94276       140.198     0.873857  1.78629
       4     28.2255      6.5212        140.272     0.838915  2.30847
       5     30.6354      6.73647       140.247     0.86363   2.8258
       6     32.1027      6.85601       140.239     0.872475  3.1542
       7     32.9709      6.89606       140.094     0.917704  3.46081
       8     33.5112      6.90753       140.076     0.931866  4.08857
       9     33.8502      6.91427       139.913     0.928974  4.25243
      10     34.0963      6.93115       139.894     0.942612  5.02607
```
**The unit of the *RMSEs* and *stds* is *℃*, The unit of the *k_time* is *s*.
k_time is the time spent on building the kernel matrix.**
