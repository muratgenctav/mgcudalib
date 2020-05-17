#ifndef MGCUBLAS_H
#define MGCUBLAS_H

namespace mgcu { namespace blas { 
    // LEVEL 1 BLAS
    namespace lvl1 {
        // DAXPY
        /**
         * Scales vector x by alpha
         * Adds the result (alpha*x) to vector y
         * y = alpha * x + y
         *
         * @param n [in] vector dimension
         * @param alpha [in] scale factor
         * @param x [in] vector x
         * @param y [in, out] vector y
         */
        void daxpy(
            const int n,
            const double alpha,
            const double * const x, 
            double * const y
        );
    } 

    // LEVEL 2 BLAS
    namespace lvl2 {
        // SPMV
        /**
         * Sparse matrix, dense vector multiplication.
         * y = alpha * (A * x) + beta * y
         * 
         * @param m [in] number of matrix rows
         * @param n [in] number of matrix columns
         * @param nnz [in] number of nonzeros
         * @param alpha [in] factor alpha
         * @param beta [in] factor beta
         * @param aVal [in] array of nonzero elements
         * @param aCol [in] array of CSR columns
         * @param aRowPtr [in] array of CSR row pointers
         * @param x [in] x vector
         * @param y [in, out] y vector
         */
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
            double * const y
        );
    }
} }

#endif