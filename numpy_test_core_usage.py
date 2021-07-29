import os
os.environ['OMP_PROC_BIND'] = 'true'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ["OPENBLAS_NUM_THREADS"] = '8' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '8' # export MKL_NUM_THREADS=6
#os.environ["VECLIB_MAXIMUM_THREADS"] = '4' # export VECLIB_MAXIMUM_THREADS=4
#os.environ["NUMEXPR_NUM_THREADS"] = '4' # export NUMEXPR_NUM_THREADS=6
import numpy as np


test_1 = np.random.rand(4, 4, 3)
test_2 = np.random.rand(4, 3, 3)

while True:
    np.matmul(test_1, test_2)
