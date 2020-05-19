#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cusparse.h"
#include "mmio.h"
#include "mgcucommon/mgcucommon.cuh"
#include "mgcublas/mgcublas.cuh"

// Pointers (host)
int * h_CooI = NULL;
int * h_CooJ = NULL;
double * h_Val = NULL;
double * h_VectIn = NULL;
double * h_VectOut = NULL;
// Pointers (device)
int * d_CooI = NULL;    
int * d_CooJ = NULL;
int * d_CsrRowPtr = NULL;
double * d_Val = NULL;
double * d_VectIn = NULL;
double * d_VectOut = NULL;
double * d_MyVectOut = NULL;
double * d_CscVal = NULL;
int * d_CscRow = NULL;
int * d_CscColPtr = NULL;

// cuSparse handle and descriptors
cusparseHandle_t handle = 0;
cusparseMatDescr_t matA = 0;

// Cleanup routine
void cleanup()
{
    if (h_CooI)         free(h_CooI);
    if (h_CooJ)         free(h_CooJ);
    if (h_Val)          free(h_Val);
    if (h_VectIn)       free(h_VectIn);
    if (h_VectOut)      free(h_VectOut);
    if (d_CooI)         cudaFree(d_CooI);
    if (d_CooJ)         cudaFree(d_CooJ);
    if (d_CsrRowPtr)    cudaFree(d_CsrRowPtr);
    if (d_Val)          cudaFree(d_Val);
    if (d_VectIn)       cudaFree(d_VectIn);
    if (d_VectOut)      cudaFree(d_VectOut);
    if (d_MyVectOut)    cudaFree(d_MyVectOut);
    if (d_CscVal)       cudaFree(d_CscVal);
    if (d_CscRow)       cudaFree(d_CscRow);
    if (d_CscColPtr)    cudaFree(d_CscColPtr);
    if (matA)       cusparseDestroyMatDescr(matA);
    if (handle)     cusparseDestroy(handle);
    cudaDeviceReset();
}

void checkErrors(cudaError_t status)
{
    if (status != cudaSuccess) {
        cleanup();
        checkCudaErrors(status);
    }
}

void checkErrors(cusparseStatus_t status, const char * msg)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cleanup();
        std::cerr << msg << std::endl;
        exit(1);
    }
}

int main(int argc, char **argv) 
{
    // Argument check
    if (argc < 2) {
        std::cerr << "Usage: spmv <filename>" << std::endl;
        cleanup();
        exit(EXIT_FAILURE);
    }
    bool colMajor = false;
    if ( (argc == 3) && (strcmp(argv[2], "-c") == 0) ) {
        colMajor = true;
    }
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "Error: No devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0) {
        std::cout << "Using device " << dev << std::endl;
        std::cout << devProps.name << "; global mem: "  
                  << devProps.totalGlobalMem << "B; compute v"
                  << devProps.major << "."
                  << devProps.minor << "; clock: "
                  << devProps.clockRate << " kHz" << std::endl;
    }

    // Read matrix in COO format
    int nnz, m, n;
    if (mm_read_unsymmetric_sparse(argv[1], &m, &n, &nnz, &h_Val, &h_CooI, &h_CooJ) != 0) {
        cleanup();
        exit(EXIT_FAILURE);
    }
    #ifdef DBG
        std::cout << "Num rows: " << m << "; Num cols: " << n << "; Num nonzeros: " << nnz << std::endl;
    #endif

    // Create random input vector
    h_VectIn = (double *) malloc((size_t) n * sizeof(double));
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        // h_VectIn[i] = rand() % RAND_MAX;
        h_VectIn[i] = 1.0f;
    }
    
    // Allocate GPU memory
    checkErrors(cudaMalloc((void **) &d_CooI, nnz * sizeof(int)));
    checkErrors(cudaMalloc((void **) &d_CooJ, nnz * sizeof(int)));
    checkErrors(cudaMalloc((void **) &d_Val, nnz * sizeof(double)));
    checkErrors(cudaMalloc((void **) &d_VectIn, n * sizeof(double)));
    checkErrors(cudaMalloc((void **) &d_VectOut, m * sizeof(double)));
    if (colMajor) {
        checkErrors(cudaMalloc((void **) &d_CsrRowPtr, (n+1) * sizeof(int)));
    } else {
        checkErrors(cudaMalloc((void **) &d_CsrRowPtr, (m+1) * sizeof(int)));
    }
        

    // Copy inputs to GPU
    checkErrors(cudaMemcpy(d_CooI, h_CooI, (size_t) nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpy(d_CooJ, h_CooJ, (size_t) nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpy(d_Val, h_Val, (size_t) nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpy(d_VectIn, h_VectIn, (size_t) n * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize cuSparse lib
    checkErrors(cusparseCreate(&handle), 
        "Error initializing CUSPARSE");

    // Convert from COO to CSR
    if (colMajor) {
        // Matrix is transposed here (rows are the cols in fact)
        checkErrors(cusparseXcoo2csr(handle, d_CooJ, nnz, n, d_CsrRowPtr, CUSPARSE_INDEX_BASE_ZERO),
            "Error converting from COO format to CSR format");
    } else {
        checkErrors(cusparseXcoo2csr(handle, d_CooI, nnz, m, d_CsrRowPtr, CUSPARSE_INDEX_BASE_ZERO),
            "Error converting from COO format to CSR format");
    }

    // Set alpha and beta
    double alpha = 1.0f;
    double beta = 0.0f;
    
    // Create and setup sparse matrix A descriptor
    checkErrors(
        cusparseCreateMatDescr(&matA),
        "Error creating sparse matrix descriptor."
    );
    cusparseSetMatIndexBase(matA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(matA, CUSPARSE_MATRIX_TYPE_GENERAL);

    // Matrix vector multiplication
    if (colMajor) {
        checkErrors(
            cusparseDcsrmv(handle, 
                CUSPARSE_OPERATION_TRANSPOSE, 
                n, m, nnz, 
                &alpha, 
                matA, 
                d_Val, 
                d_CsrRowPtr, 
                d_CooI,
                d_VectIn, 
                &beta, 
                d_VectOut),
            "Error during multiplication"
        );
    } else {
        checkErrors(
            cusparseDcsrmv(handle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                m, n, nnz, 
                &alpha, 
                matA, 
                d_Val, 
                d_CsrRowPtr, 
                d_CooJ,
                d_VectIn, 
                &beta, 
                d_VectOut),
            "Error during multiplication"
        );
    }

    // Copy the result to host
    h_VectOut = (double *) malloc((size_t) m * sizeof(double));
    checkErrors(cudaMemcpy(h_VectOut, d_VectOut, (size_t) m * sizeof(double), cudaMemcpyDeviceToHost));

    #ifdef DBG
        for (int i = 0; i < m-1; i++) {
            std::cout << h_VectOut[i] << ", ";
        }
        std::cout << h_VectOut[m-1] << std::endl;
    #endif

    // Using my SpMv algorithm
    checkErrors(cudaMalloc((void **) &d_MyVectOut, m * sizeof(double)));
    if (colMajor) {
        // First convert to csc in order to obtain actual rows
        // (i.e. transpose the matrix)
        checkErrors(cudaMalloc((void **) &d_CscVal, nnz * sizeof(double)));
        checkErrors(cudaMalloc((void **) &d_CscRow, nnz * sizeof(int)));
        checkErrors(cudaMalloc((void **) &d_CscColPtr, (m+1) * sizeof(int)));
        checkErrors(
            cusparseDcsr2csc(handle, 
                n, m, nnz,
                d_Val, d_CsrRowPtr, 
                d_CooI, d_CscVal,
                d_CscRow, d_CscColPtr, 
                CUSPARSE_ACTION_NUMERIC, 
                CUSPARSE_INDEX_BASE_ZERO),
                "Error converting from CSR to CSC."
        );
        mgcu::blas::lvl2::spmv(
            m,n,nnz,
            alpha, beta,
            d_CscVal,
            d_CscRow,
            d_CscColPtr,
            d_VectIn,
            d_MyVectOut);
    }
    else {
        mgcu::blas::lvl2::spmv(
            m,n,nnz,
            alpha, beta,
            d_Val,
            d_CooJ,
            d_CsrRowPtr,
            d_VectIn,
            d_MyVectOut);
    }

    // Copy the result to host
    checkErrors(cudaMemcpy(h_VectOut, d_MyVectOut, (size_t) m * sizeof(double), cudaMemcpyDeviceToHost));

    #ifdef DBG
        for (int i = 0; i < m-1; i++) {
            std::cout << h_VectOut[i] << ", ";
        }
        std::cout << h_VectOut[m-1] << std::endl;
    #endif

    cleanup();
    return 0;
}

