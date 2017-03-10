# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:35:42 2016

@author: shiwu_001
"""

from config import DTYPE
from numpy import int32
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
from pycuda.tools import context_dependent_memoize

"""
    Elementwise Kernel
"""
# r = a
seta = ElementwiseKernel("float a, float *r",
                         "r[i] = a",
                         "kernel_seta")

# r = x
setx = ElementwiseKernel("float *x, float *r",
                         "r[i] = x[i]",
                         "kernel_setx")

# r = ax + b
axpb = ElementwiseKernel("float a, float *x, float b, float *r",
                         "r[i] = a*x[i] + b",
                         "kernel_axpb")

# r = sqr(x)
sqrx = ElementwiseKernel("float *x, float *r",
                         "r[i] = x[i]*x[i]",
                         "kernel_sqrx")

# r = sqrt(x)
sqrtx = ElementwiseKernel("float *x, float *r",
                          "r[i] = sqrt(x[i])",
                          "kernel_sqrtx")

# r = x / (y + eps)
xdivyeps = ElementwiseKernel("float *x, float *y, float eps, float *r",
                             "r[i] = x[i] / (y[i] + eps)",
                             "kernel_xdivyeps")

# r = ax + by
axpby = ElementwiseKernel("float a, float *x, float b, float *y, float *r",
                          "r[i] = a*x[i] + b*y[i]",
                          "kernel_axpby")

# r = ax + by + cz 
axpbypcz = ElementwiseKernel(
    "float a, float *x, float b, float *y, float c, float *z, float *r",
    "r[i] = a*x[i] + b*y[i] + c*z[i]",
    "kernel_axpbypcz")

# r = pos(x) != pos(y) w/ pos(x[i]) = x[i] > 0
xorpos = ElementwiseKernel("float *x, float *y, float *r",
                          "r[i] = (x[i]>0) != (y[i]>0)",
                          "kernel_xorpos")

# r[i] = 1. if x[i] > a;
#        0. if x[i] < -a;
#        [0,1] otherwise
softlinear = ElementwiseKernel("float a, float *x, float *r",
                               "r[i] = (x[i] > a) ? float(1.) : ((a == float(0.) || x[i] <= -a) ? float(0.) : (0.5 * x[i] / a + 0.5))",
                               "kernel_softlinear", )

# r[i] = sigmoid(x[i] / (a + eps))
softsigmoid = ElementwiseKernel("float a, float *x, float *r",
                                "r[i] = 1. / (1. + exp(-x[i] / (a + 1e-8)))",
                                "kernel_softsigmoid",
                                preamble="#include <cmath>")

# r[i] = x[i] * y[i]
eltmul = ElementwiseKernel("float *x, float *y, float *r",
                           "r[i] = x[i]*y[i]",
                           "kernel_eltmul")

# r[i] = clip(x[i], a, b), x[i] \in [a, b]
clipab = ElementwiseKernel("float *x, float a, float b, float *r",
                           "r[i] = (x[i] > a) ? a : ((x[i] < b) ? b : x[i])",
                           "kernel_clipab")

"""
    Reduction Kernel
"""
# (r) = sum(abs(x))
@context_dependent_memoize
def get_sumabs_kernel():
    return ReductionKernel(DTYPE, neutral="0.",
                         reduce_expr="a+b",
                         map_expr="abs(x[i])",
                         arguments="float *x",
                         name="kernel_sumabs")

# (r) = sum(sqr(x))
@context_dependent_memoize
def get_sumsqr_kernel():
    return ReductionKernel(DTYPE, neutral="0.",
                         reduce_expr="a+b",
                         map_expr="x[i]*x[i]",
                         arguments="float *x",
                         name="kernel_sumsqr")

# (r) = sum(pos(x) != pos(y)) w/ pos(x[i]) = x[i] > 0
@context_dependent_memoize
def get_sumxorpos_kernel():
    return ReductionKernel(int32, neutral="0",
                            reduce_expr="a+b",
                            arguments="float *x, float *y",
                            map_expr="(x[i] > 0) != (y[i] > 0)",
                            name="kernel_sumxorpos")

def sumabs(x, stream=None, allocator=None):
    krnl = get_sumabs_kernel()
    return krnl(x, stream=stream, allocator=allocator)

def sumsqr(x, stream=None, allocator=None):
    krnl = get_sumsqr_kernel()
    return krnl(x, stream=stream, allocator=allocator)
    
def sumxorpos(x, y, stream=None, allocator=None):
    krnl = get_sumxorpos_kernel()
    return krnl(x, y, stream=stream, allocator=allocator)
    
#if __name__ == "__main__":
#    import pycuda.autoinit
#    import pycuda.gpuarray as garr
#    import numpy as np
#    a = np.arange(5, dtype=DTYPE)
#    print a
#    a_gpu = garr.to_gpu(a)
#    b_gpu = a_gpu.copy()
#    print "a =", a_gpu.get()
#    print "b =", b_gpu.get()
#    print "sumxorpos", sumxorpos(a_gpu, b_gpu)
#    a_gpu -= 2
#    print "a =", a_gpu.get()
#    print "b =", b_gpu.get()
#    print "sumxorpos", sumxorpos(a_gpu, b_gpu)
#    soft_a = np.float32(0)
#    softlinear(soft_a, a_gpu, b_gpu)
#    print "softlinear(%f, a) =" % soft_a, b_gpu