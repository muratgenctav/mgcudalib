#include "mgcucommon/mgcucommon.cuh"
#include "mgcublas/mgcublas.cuh"
#include "mgcuutils/mgcuutils.cuh"

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// LEVEL-1 BLAS FUNCTIONS
namespace mgcu { namespace blas {
    // LEVEL 1 BLAS
    namespace lvl1 { 
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
    }

    // LEVEL 2 BLAS
    namespace lvl2 { 
        namespace helpers {
            // GATHER_RESULT helper kernel for SPMV
            template<typename T>
            __global__
            void gatherMvResult(
                T * const d_out,
                const T * const d_in,
                const unsigned int * const d_pos,
                const unsigned int nElements)
            {
                unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

                // boundary check
                if (idx >= nElements) return;

                // select the input element at specified position
                unsigned int pos = d_pos[idx];
                d_out[idx] = d_in[pos];
                if ( (idx > 0) && (pos == d_pos[idx-1]) ){
                    // The row is empty
                    d_out[idx] = (T) 0;
                }
            }
        }

        // SPMV function
        void spmv(
            const int m,
            const int n,
            const int nnz,
            const double alpha,
            const double beta,
            const double * const aVal,
            const int * const aCol,
            const int * const aRowPtr,
            const double * const x,
            double * const y)
        {

            // allocate device memory for temporary storage
            double * tempProdArr;
            checkCudaErrors(cudaMalloc(&tempProdArr, nnz * sizeof(double)));

            // gather coefficients from input vector
            mgcu::utils::gather<double>(tempProdArr, x, aCol, nnz);

            // make element-wise multiplication
            mgcu::utils::map<double>(tempProdArr, aVal, nnz, mgcu::utils::MAP_MULTIPLICATION);

            // compute row-wise accumulation
            mgcu::utils::accumulateSeg<double>(tempProdArr, nnz, aRowPtr, m, mgcu::utils::SCAN_INCLUSIVE);

            // compute row-end positions
            int * aRowEndPtr;
            checkCudaErrors(cudaMalloc(&aRowEndPtr, m * sizeof(int)));
            mgcu::utils::shift<int>(aRowEndPtr, aRowPtr, m, -1, nnz);
            mgcu::utils::map<int>(aRowEndPtr, 1, m, mgcu::utils::MAP_SUBTRACTION);

            // gather values at the row ends to obtain the result of A*x
            double * tempResArr;
            checkCudaErrors(cudaMalloc(&tempResArr, m * sizeof(double)));
            int numThreads = MAX_THREADS;
            int numBlocks = ceil((float) m / numThreads);
            helpers::gatherMvResult(tempResArr, tempProdArr, aRowEndPtr, m);

            // scale y using beta (y = beta*y)
            mgcu::utils::map<double>(y, beta, m, mgcu::utils::MAP_MULTIPLICATION);

            // apply daxpy to obtain the result of y = alpha*A*x + beta*y 
            mgcu::blas::lvl1::daxpy(m, alpha, tempResArr, y);

            // free allocated device memory
            checkCudaErrors(cudaFree(tempProdArr));
            checkCudaErrors(cudaFree(tempResArr));
            checkCudaErrors(cudaFree(aRowEndPtr));
        }
    }
} }
