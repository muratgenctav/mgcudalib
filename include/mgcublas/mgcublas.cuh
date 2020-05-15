#ifndef MGCUBLAS_H
#define MGCUBLAS_H

namespace mgcu { namespace blas { namespace lvl1 {
    // DAXPY
    /**
     * Scales vector x by alpha
     * Adds the result (a*x) to vector y
     * y = a * x + y
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
} } }

#endif