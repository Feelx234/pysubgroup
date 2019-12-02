import numpy as np
import cupy as cp
import time
from numba import njit
from numba import prange

from numba import int32,uint8,bool_

@njit(int32(bool_[:],bool_[:]),parallel=True)
def and_reduce(x,y):
    count = int(0)
    assert(len(x) == len(y))
    for i in prange(len(x)):
        if x[i] & y[i]:
            count+=1
    return count

@njit(int32(uint8[:],uint8[:]),parallel=True)
def and_reduce2(x,y):
    count = int(0)
    assert(len(x) == len(y))
    for i in prange(len(x)):
        tmp = x[i] & y[i]
        if tmp & 1:
            count+=1
        if tmp & 2:
            count+=1
        if tmp & 4:
            count+=1
        if tmp & 8:
            count+=1
        if tmp & 16:
            count+=1
        if tmp & 32:
            count+=1
        if tmp & 64:
            count+=1
        if tmp & 128:
            count+=1
    return count

@njit(int32(uint8[:],uint8[:]),parallel=True)
def and_reduce3(a, b):
    count = int(0)
    assert(len(a) == len(b))
    for i in prange(len(a)):
        x = a[i] & b[i]
        count += (x & 1) + (x & 2) + (x & 4) + (x & 8) + (x & 16) + (x & 32) + (x & 64) + (x & 128)
    return count

@njit(parallel=True)
def fill_array(arr):
    for i in prange(len(arr)):
        arr[i] = (np.random.rand(1)>0.3)[0]

@njit(parallel=True)
def fill_array2(arr):
    for i in prange(len(arr)):
        arr[i] = np.random.rand(1)[0]

def time_and_reduce(fun, arr1, arr2, title):
    z = fun(arr1, arr2)
    start_time = time.time()
    z = fun(arr1, arr2)
    delta = time.time() - start_time
    print(title+" --- %s seconds ---" % (delta))
    #print(z)
    return delta


n=int(10**9)
#target = np.empty(n, dtype=bool)
#fill_array2(target)
x_cpu = np.empty(n, dtype=bool)
y_cpu = np.empty(n, dtype=bool)
fill_array(y_cpu)
fill_array(x_cpu)
x_gpu = cp.asarray(x_cpu)
y_gpu = cp.asarray(y_cpu)
x_dense = np.packbits(x_cpu)
y_dense = np.packbits(y_cpu)

time_and_reduce(and_reduce, x_cpu, y_cpu, 'numba_normal')
time_and_reduce(and_reduce2, x_dense, y_dense, 'numba_dense1')
time_and_reduce(and_reduce3, x_dense, y_dense, 'numba_dense2')

start_time = time.time()
z = np.count_nonzero(np.logical_and(x_cpu, y_cpu))
print('numpy_extern'+" --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
z = np.count_nonzero(x_cpu[y_cpu])
print('numpy_index '+" --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
x_cpu &= y_cpu
z = np.count_nonzero(x_cpu)
print('numpy_inplac'+" --- %s seconds ---" % (time.time() - start_time))



cp.cuda.Stream.null.synchronize()
start_time = time.time()
z_gpu = cp.count_nonzero(np.logical_and(x_gpu, y_gpu))
cp.cuda.Stream.null.synchronize()
print("cupy_extern_c--- %s seconds ---" % (time.time() - start_time))


cp.cuda.Stream.null.synchronize()
start_time = time.time()
z_gpu = cp.sum(np.logical_and(x_gpu, y_gpu))
cp.cuda.Stream.null.synchronize()
print("cupy_extern_s--- %s seconds ---" % (time.time() - start_time))


cp.cuda.Stream.null.synchronize()
start_time = time.time()
x_gpu &= y_gpu
z_gpu = cp.sum(x_gpu)
cp.cuda.Stream.null.synchronize()
print("cupy_inplace --- %s seconds ---" % (time.time() - start_time))


x_gpu = cp.asarray(x_dense)
y_gpu = cp.asarray(y_dense)


cp.cuda.Stream.null.synchronize()
start_time = time.time()
z_gpu = cp.sum(cp.logical_and(x_gpu, y_gpu))
cp.cuda.Stream.null.synchronize()
print("cupy_8bit  --- %s seconds ---" % (time.time() - start_time))