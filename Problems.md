# About graph kenrels.

## (Random walk) Sylvester equation kernel.

### ImportError: cannot import name 'frange' from 'matplotlib.mlab'

You are using an outdated `control` with a recent `matplotlib`. `mlab.frange` was removed in `matplotlib-3.1.0`, and `control` removed the call in `control-0.8.2`.

Update your `control` package.

### Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.

The Intel Math Kernel Library (MKL) is missing or not properly set. I assume the MKL is required by `control` module.

Install MKL. Then add the following to your path:

```
export PATH=/opt/intel/bin:$PATH

export LD_LIBRARY_PATH=/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
```
