#include "mgcucommon/mgcucommon.cuh"
#include "mgcublas/mgcublas.cuh"
#include "mgcuutils/mgcuutils.cuh"

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// LEVEL-1 BLAS FUNCTIONS
namespace mgcu { namespace blas { namespace lvl1 { 
    namespace kernels {
        // DAXPY kernel
        __global__ void daxpy(
            const int n,
            const double alpha,
            const double * const x, 
            double * const y) 
        {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

            // check boundary
            if (i >= n) return;

            // compute
            y[i] = alpha * x[i] + y[i];
        }
    } 

    // DAXPY function
    void daxpy(const int n, const double alpha, const double * const x, double * const y)
    {
        int numThreads = MAX_THREADS;
        int numBlocks = ceil((float) n / numThreads);

        kernels::daxpy<<<numBlocks, numThreads>>>(n, alpha, x, y);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }
} } }
