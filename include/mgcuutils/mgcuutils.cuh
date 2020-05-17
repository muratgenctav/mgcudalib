#ifndef MGCUUTILS_H
#define MGCUUTILS_H

namespace mgcu { namespace utils {
    
    
    // GATHER
    /**
     * Gathers selected elements of an array arrIn into a new array arrOut.
     *
     * @param arrOut [out] Array of gathered elements.
     * @param arrIn [in] Input array.
     * @param arrIdx [in] Indices of input array elements to be gathered.
     * @param numIdxs [in] Number of input array elements to be gathered.
     */ 
    template<typename T>
    void gather(
        T * const arrOut,
        const T * const arrIn,
        const int * arrIdx,
        const int numIdxs
    );

    // SCATTER constant
    /**
     * Scatters (adds) constant val onto arrOut in dense format.
     * 
     * @param arrOut [out] Output array
     * @param arrLen [in] Length of arrOut
     * @param val [in] Constant value to scatter
     * @param arrIdx [in] Positions of output array where val is going to be placed
     * @param numIdxs [in] Number of positions where val is going to be placed
     */
    template<typename T>
    void scatterConst(
        T * const arrOut,
        const int arrLen,
        const T val,
        const int * arrIdx,
        const int numIdxs
    );

    //SHIFT
    /**
     * Shifts elements of an array arrIn by number of positions "offset"
     * and returns shifted array in arrOut. The direction of the shift is 
     * determined by the sign of offset (+ Right / - Left).
     *
     * @param arrOut [out] Shifted array.
     * @param arrIn [in] Input array.
     * @param arrLen [in] Array length.
     * @param offset [in] Number of positions input array elements to be shifted.
     * @param fillIn [in] Fill-in value.
     */ 
    template<typename T>
    void shift(
        T * const arrOut, 
        const T * const arrIn, 
        const int arrLen, 
        int offset,
        T fillIn
    );

    // Segmented SHIFT with segment heads
    /**
     * Shifts elements of a segmented array arrIn by number of positions
     * "offset" and returns shifted array in arrOut. The direction of the 
     * shift is determined by the sign of offset (+ Right / - Left).
     * Segment heads are given in array segHeads.
     * 
     * @param arrOut [out] Shifted array.
     * @param arrIn [in] Input array.
     * @param arrLen [in] Array length.
     * @param offset [in] Number of positions input array elements to be shifted.
     * @param fillIn [in] Fill-in value.
     * @param segHeads [in] Head positions of segment heads.
     * @param numSegments [in] Number of segments.
     */
    template<typename T>
    void shiftSeg(
        T * const arrOut, 
        const T * const arrIn, 
        const int arrLen, 
        int offset, 
        T fillIn, 
        const int * const segHeads, 
        const int numSegments
    );

    // Segmented SHIFT function with segment ids
    /**
     * Shifts elements of a segmented array arrIn by number of positions
     * "offset" and returns shifted array in arrOut. The direction of the 
     * shift is determined by the sign of offset (+ Right / - Left).
     * Segment ids are given in array segIds.
     * 
     * @param arrOut [out] Shifted array.
     * @param arrIn [in] Input array.
     * @param segHeads [in] Segment id of each array element.
     * @param arrLen [in] Array length.
     * @param offset [in] Number of positions input array elements to be shifted.
     * @param fillIn [in] Fill-in value.
     */
    template<typename T>
    void shiftSeg(
        T * const arrOut, 
        const T * const arrIn, 
        const int * segIds, 
        const int arrLen, 
        int offset, 
        T fillIn
    );

    // MAP
    /**
     * List of map operations
     */
    typedef enum
    {
        MAP_SUMMATION,          /**< += operation. */
        MAP_SUBTRACTION,        /**< -= operation. */
        MAP_MULTIPLICATION,     /**< *= operation. */
        MAP_DIVISION            /**< /= operation. */
    } map_t;

    /**
     * Applies the following element-wise operation:
     * lhsArr[i] <op> rhsArr[i], i=0..arrLen
     * 
     * @param lhsArr [in, out] left operand array.
     * @param rhsArr [in] right operand array.
     * @param arrLen [in] length of both arrays.
     * @param op [in] specifies operation.
     * @see {@link map_t} 
     */
    template<typename T>
    void map(
        T * const lhsArr, 
        const T * const rhsArr, 
        const int arrLen, 
        map_t op
    );

    /**
     * Applies the following element-wise operation:
     * lhsArr[i] <op> rhsConst, i=0..arrLen
     * 
     * @param lhsArr [in, out] left operand array.
     * @param rhsConst [in] right operand value.
     * @param arrLen [in] length of both arrays.
     * @param op [in] specifies operation.
     * @see {@link map_t} 
     */
    template<typename T>
    void map(
        T * const lhsArr, 
        const T rhsVal, 
        const int arrLen, 
        map_t op
    );

    // SCAN
    /**
     * List of scan types
     */
    typedef enum
    {
        SCAN_EXCLUSIVE, /**< Exclusive scan. */
        SCAN_INCLUSIVE  /**< Inclusive scan. */
    } scan_t;

    /**
     * Accumulates an array onto itself.
     * 
     * Implements Hillis&Steele scan algorithm.
     * 
     * @param arr [in,out] Array to scan.
     * @param arrLen [in] Length of arr.
     * @param scanType [in] Type of scan.
     * @see {@link scan_t} 
     */
    template<typename T>
    void accumulate(
        T * const arr, 
        const int arrLen, 
        const scan_t scanType
    );

    /**
     * Accumulates a segmented array onto itself.
     * 
     * Implements Hillis&Steele scan algorithm.
     * 
     * @param arr [in,out] Array to scan
     * @param arrLen [in] Length of arr
     * @param segHeads [in] Head positions of segments
     * @param numSegments [in] Number of segments
     * @param scanType [in] Type of scan
     * @see {@link scan_t} 
     */
    template<typename T>
    void accumulateSeg(
        T * const arr, 
        const int arrLen, 
        const int * const segHeads, 
        const int numSegments, 
        const scan_t scanType
    );
    
} }

// include implementations
#include "mgcuutils_impl.cuh"

#endif