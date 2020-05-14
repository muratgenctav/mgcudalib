#ifndef MGCUUTILS_IMPL_H
#define MGCUUTILS_IMPL_H

#include "mgcucommon/mgcucommon.cuh"

#include <cmath>

namespace mgcu::utils::kernels
{
    // GATHER kernel
    template<typename T>
    __global__
    void gather(
        T * const d_out,
        const T * const d_in,
        const int * const d_pos,
        const int nElements)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // boundary check
        if (idx >= nElements) return;

        // select the input element at specified position
        int pos = d_pos[idx];
        d_out[idx] = d_in[pos];
    }

    // SCATTER constant kernel
    template<typename T>
    __global__
    void scatterConst(
        T * const arrOut,
        const int arrLen,
        const T val,
        const int * arrIdx,
        const int numIdxs)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < numIdxs)
        {
            int pos = arrIdx[i];
            if (pos < arrLen) {
                atomicAdd(&arrOut[pos], val);
            }
        }
    }

    // SHIFT kernel
    template<typename T>
    __global__ 
    void shift(
        T * const d_out, 
        const T * const d_in, 
        const int size, 
        int n,
        T padVal)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // boundary check
        if (i >= size) return;

        int j = i - n;

        if (j >= 0)
        {
            d_out[i] = d_in[j];
        }
        else
        {
            // pad with padding value
            d_out[i] = padVal;
        }
    }

    // Segmented SHIFT kernel
    template<typename T>
    __global__
    void shiftSeg(
        T * const d_out, 
        const T * const d_in,
        const unsigned int * const d_seg,
        const unsigned int size, 
        int n,
        T padVal)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // boundary check
        if (i >= (int) (size & INT_MAX)) return;

        int j = i - n;

        if ( (j >= 0) && (d_seg[i] == d_seg[j]))
        {
            d_out[i] = d_in[j];
        }
        else
        {
            // pad with paddin value
            d_out[i] = padVal;
        }
    }

    // SCAN kernel
    template<typename T>
    __global__ void scan(
        T * const d_out, 
        const T * const d_in, 
        const int size, 
        const int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // boundary check
        if (i >= size) return;
        
        if (n <= i)
        {
            int j = i - n;
            d_out[i] = d_in[i] + d_in[j];
        }
        else
        {
            // copy original value
            d_out[i] = d_in[i];
        }
    }

    // Segmented SCAN kernel
    template<typename T>
    __global__ void segscan(
        T * const d_out, 
        const T * const d_in,
        const int * d_seg,
        const int size, 
        const int n)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        // boundary check
        if (i >= size) return;
        
        int j = i - n;

        if ((j >= 0) && (d_seg[i] == d_seg[j]))
        {
            d_out[i] = d_in[i] + d_in[j];
        }
        else
        {
            // copy original value
            d_out[i] = d_in[i];
        }
    }
}

namespace mgcu::utils
{
    // GATHER function
    template<typename T>
    void gather(T * const arrOut, const T * const arrIn, const int * const arrIdx, const int numIdxs)
    {
        int numThreads = MAX_THREADS;
        int numBlocks = ceil((float) numIdxs / numThreads);

        // kernel call
        kernels::gather<T><<<numBlocks, numThreads>>>(arrOut, arrIn, arrIdx, numIdxs);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    // SCATTER constant function
    template<typename T>
    void scatterConst(T * const arrOut, const int arrLen, const T val, const int * arrIdx, const int numIdxs)
    {
        int numThreads = MAX_THREADS;
        int numBlocks = ceil((float) numIdxs / numThreads);

        // kernel call
        kernels::scatterConst(arrOut, arrLen, val, arrIdx, numIdxs);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    // SHIFT function
    template<typename T>
    void shift(T * const arrOut, const T * const arrIn, const int arrLen, int offset, T fillIn)
    {
        int numThreads = MAX_THREADS;
        int numBlocks = ceil((float) arrLen / numThreads);

        // kernel call
        kernels::shift<T><<<numBlocks, numThreads>>>(arrOut, arrIn, arrLen, offset, fillIn);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    // Segmented SHIFT function
    // with segment heads
    template<typename T>
    void shiftSeg(T * const arrOut, const T * const arrIn, const int arrLen, int offset, T fillIn, 
        const int * const segHeads, const int numSegments)
    {
        int numThreads = MAX_THREADS;
        int numBlocks = ceil((float) arrLen / numThreads);

        // allocate device memory for segment ids
        int * segIds;
        checkCudaErrors(cudaMalloc(&segIds, arrLen * sizeof(int)));
        checkCudaErrors(cudaMemset(segIds, 0, arrLen * sizeof(int)));

        // first mark segment heads in the segments array
        scatterConst<int>(segIds, arrLen, 1, segHeads, numSegments);

        // compute segment ids using basic scan
        accumulate<int>(segIds, arrLen, SCAN_INCLUSIVE);

        // kernel call
        kernels::shiftSeg<T><<<numBlocks, numThreads>>>(arrOut, arrIn, segIds, arrLen, offset, fillIn);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        cudaFree(segIds);
    }

    // Segmented SHIFT function
    // with segment ids
    template<typename T>
    void shiftSeg(T * const arrOut, const T * const arrIn, const int * segIds, const int arrLen, int offset, T fillIn)
    {
        int numThreads = MAX_THREADS;
        int numBlocks = ceil((float) arrLen / numThreads);

        // kernel call
        kernels::shiftSeg<T><<<numBlocks, numThreads>>>(arrOut, arrIn, segIds, arrLen, offset, fillIn);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    // ACCUMULATE function
    template<typename T>
    void accumulate(T * const arr, const int arrLen, const scan_t scanType)
    {
        int numThreads = MAX_THREADS;
        int numBlocks = ceil((float) arrLen / numThreads);

        // allocate device memory for temporary array
        T * tempArr;
        checkCudaErrors(cudaMalloc(&tempArr, arrLen * sizeof(T)));

        // rdy indicates if the output is ready on arr
        bool rdy = true;

        // shift array if exclusive scan requested
        if (scanType == SCAN_EXCLUSIVE)
        {
            shift(tempArr, arr, arrLen, 1, 0);
            rdy = false;
        }

        // run scan steps
        int n = 1;
        while (n < arrLen)
        {
            if (rdy)
            {
                kernels::scan<T><<<numBlocks, numThreads>>>(tempArr, arr, arrLen, n);
            }
            else
            {
                kernels::scan<T><<<numBlocks, numThreads>>>(arr, tempArr, arrLen, n);
            }
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

            rdy = !rdy;
            n <<= 1;
        }

        // make sure output is in arr
        if (!rdy)
        {
            checkCudaErrors(cudaMemcpy(arr, tempArr, arrLen * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        cudaFree(tempArr);
    }

    // Segmented ACCUMULATE function
    template<typename T>
    void accumulateSeg(T * const arr, const int arrLen, const int * const segHeads, const int numSegments, const scan_t scanType)
    {
        int numThreads = MAX_THREADS;
        int numBlocks;

        // Preprocessing

        // allocate device memory for segment ids
        int * segIds;
        checkCudaErrors(cudaMalloc(&segIds, arrLen * sizeof(int)));
        checkCudaErrors(cudaMemset(segIds, 0, arrLen * sizeof(int)));

        // first mark segment heads in the segments array
        scatterConst<int>(segIds, arrLen, 1, segHeads, numSegments);

        // compute segment ids using basic scan
        accumulate<int>(segIds, arrLen, SCAN_INCLUSIVE);

        // Compute segmented scan!!!!!!!!!!!!!!
        numBlocks = ceil((float) arrLen / numThreads);

        // allocate device memory for temporary array
        T * tempArr;
        checkCudaErrors(cudaMalloc(&tempArr, arrLen * sizeof(T)));

        // rdy indicates if the output is ready on arr
        bool rdy = true;

        // shift array if exclusive scan requested
        if (scanType == SCAN_EXCLUSIVE)
        {
            shiftSeg<T>(tempArr, arr, segIds, arrLen, 1, 0);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
            rdy = false;
        }

        // run scan steps
        unsigned int n = 1;
        while (n < arrLen)
        {
            if (rdy)
            {
                kernels::scanSeg<T><<<numBlocks, numThreads>>>(tempArr, arr, segIds, arrLen, n);
            }
            else
            {
                kernels::scanSeg<T><<<numBlocks, numThreads>>>(arr, tempArr, segIds, arrLen, n);
            }
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

            rdy = !rdy;
            n <<= 1;
        }

        // make sure output is in arr
        if (!rdy)
        {
            checkCudaErrors(cudaMemcpy(arr, tempArr, arrLen * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        checkCudaErrors(cudaFree(segIds));
        checkCudaErrors(cudaFree(tempArr));
    }
}

#endif