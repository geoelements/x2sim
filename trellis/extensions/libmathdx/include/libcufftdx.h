// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

// libmathdx's API is subject to change.
// Please contact Math-Libs-Feedback@nvidia.com for usage feedback.

#ifndef MATHDX_CUFFTDX_H
#define MATHDX_CUFFTDX_H

#include <stddef.h>

#include "libcommondx.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * @brief A cuFFTDx descriptor
 *
 * Equivalent to `using FFT = ...` in cuFFTDx CUDA C++.
 */
typedef long long int cufftdxDescriptor;

/**
 * @brief Type of cufftDx API
 *
 * Handling problems with input in register or in shared memory buffers.
 *
 * Check cufftdx::execute method documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/methods.html#block-execute-method)
 */
typedef enum cufftdxApi_t {
    /**
     * Input-output is in thread-local memory. Shared memory is used as scratch for block execution.
     * Function API is defined by its signature.
     * Block execution: `void (value_type* rmem, char* smem)`.
     * Thread execution: `void (value_type* rmem)`
     *     - `rmem` is a pointer to an array of value_type values. See \ref CUFFTDX_TRAIT_VALUE_TYPE
     *     - `smem` is a pointer to \ref CUFFTDX_TRAIT_SHARED_MEMORY_SIZE bytes in shared memory.
     *
     * The function is `extern "C"` and the symbol name can be queried using \ref CUFFTDX_TRAIT_SYMBOL_NAME
     */
    CUFFTDX_API_LMEM = 0,
    /**
     * Input-output is in shared memory
     * Function API is defined by its signature: `void (value_type* smem)`
     *     - `smem` is a pointer to a shared memory array of value_type values. See \ref CUFFTDX_TRAIT_VALUE_TYPE.
     *
     * The function is `extern "C"` and the symbol name can be queried using \ref CUFFTDX_TRAIT_SYMBOL_NAME
     */
    CUFFTDX_API_SMEM = 1,
} cufftdxApi;

/**
 * @brief Type of computation data
 *
 * Check cufftdx::Type operator documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/methods.html#block-execute-method)
 */
typedef enum cufftdxType_t {
    /** Complex-to-complex FFT */
    CUFFTDX_TYPE_C2C = 0,
    /** Real-to-complex FFT */
    CUFFTDX_TYPE_R2C = 1,
    /** Complex-to-real FFT */
    CUFFTDX_TYPE_C2R = 2,
} cufftdxType;

/**
 * @brief FFT direction
 *
 * Check cufftdx::Direction operator documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/operators.html#direction-operator)
 */
typedef enum cufftdxDirection_t {
    /** Forward FFT */
    CUFFTDX_DIRECTION_FORWARD = 0,
    /** Inverse FFT */
    CUFFTDX_DIRECTION_INVERSE = 1,
} cufftdxDirection;

/**
 * @brief Complex data layout
 *
 * Check cufftdx layout documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/methods.html#complex-element-layouts)
 */
typedef enum cufftdxComplexLayout_t {
    /** Natural complex layout (N/2+1 complex values) */
    CUFFTDX_COMPLEX_LAYOUT_NATURAL = 0,
    /** Packed complex layout (N/2 complex values) */
    CUFFTDX_COMPLEX_LAYOUT_PACKED = 1,
    /** Full complex layout (N complex values) */
    CUFFTDX_COMPLEX_LAYOUT_FULL = 2
} cufftdxComplexLayout;

/**
 * @brief Real data mode
 *
 * Check cufftdx real data mode documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/methods.html#real-element-layouts)
 */
typedef enum cufftdxRealMode_t {
    /** Normal real mode (N real values) */
    CUFFTDX_REAL_MODE_NORMAL = 0,
    /** Fold real mode (N/2-1 complex values) */
    CUFFTDX_REAL_MODE_FOLDED = 1,
} cufftdxRealMode;

/**
 * @brief cufftDx operators
 *
 * The set of supported cufftDx operators.
 *
 * Check cufftDx description operators documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/operators.html#description-operators)
 *
 * Check cufftDx execution operators documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/operators.html#execution-operators)
 */
typedef enum cufftdxOperatorType_t {
    /** Operator data type: long long int.
     * Expected content: >= 1.
     * Operator definition: required */
    CUFFTDX_OPERATOR_SIZE = 0,
    /** Operator data type: \ref cufftdxDirection_t.
     * Operator definition: required */
    CUFFTDX_OPERATOR_DIRECTION = 1,
    /** Operator data type: \ref cufftdxType_t.
     * Operator definition: optional */
    CUFFTDX_OPERATOR_TYPE = 2,
    /** Operator data type: \ref commondxPrecision_t.
     * Operator definition: required */
    CUFFTDX_OPERATOR_PRECISION = 3,
    /** Operator data type: long long int.
     * Expected content: 700 (Volta), 800 (Ampere), ....
     * Operator definition: required */
    CUFFTDX_OPERATOR_SM = 4,
    /** Operator data type: \ref commondxExecution_t.
     * Operator definition: required */
    CUFFTDX_OPERATOR_EXECUTION = 5,
    /** Operator data type: long long int.
     * Expected content: >= 0.
     * Operator definition: optional */
    CUFFTDX_OPERATOR_FFTS_PER_BLOCK = 6,
    /** Operator data type: long long int.
     * Expected content: >= 0.
     * Operator definition: optional */
    CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD = 7,
    /** Operator data type: long long int * 3.
     * Expected content: <x, y, z> block dimensions.
     * Operator definition: optional */
    CUFFTDX_OPERATOR_BLOCK_DIM = 8,
    /** Operator data type: \ref cufftdxComplexLayout_t followed by \ref cufftdxRealMode_t.
     * Operator definition: optional */
    CUFFTDX_OPERATOR_REAL_FFT_OPTIONS = 9,
    /** Operator data type: \ref cufftdxApi_t.
     * Operator definition: required */
    CUFFTDX_OPERATOR_API = 10,
} cufftdxOperatorType;

/**
 * @brief cufftDx configuration knobs
 *
 * The set of supported knobs used for accessing cufftDx operator's
 * performance configuration details.
 *
 * Check cufftDx Execution Traits documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/traits.html#execution-traits)
 */
typedef enum cufftdxKnobType_t {
    /** Elements per thread */
    CUFFTDX_KNOB_ELEMENTS_PER_THREAD = 0,
    /** FFTs per block */
    CUFFTDX_KNOB_FFTS_PER_BLOCK = 1,
} cufftdxKnobType;

/**
 * @brief cufftDx traits
 *
 * The set of supported types of traits that can be accessed from finalized sources
 * that use cufftDx.
 *
 * Check cufftDx Execution Thread Traits documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/traits.html#thread-traits)
 *
 * Check cufftDx Execution Block Traits documentation for more details
 * (https://docs.nvidia.com/cuda/cufftdx/api/traits.html#block-traits)
 */
typedef enum cufftdxTraitType_t {
    /** Trait data type: \ref commondxValueType_t.
     * Expected content: complex type of the underlying data used to compute the FFT */
    CUFFTDX_TRAIT_VALUE_TYPE = 0,
    /** Trait data type: \ref commondxValueType_t.
     * Expected content: type of the underlying data used as input for the FFT */
    CUFFTDX_TRAIT_INPUT_TYPE = 1,
    /** Trait data type: \ref commondxValueType_t.
     * Expected content: type of the underlying data used as output for the FFT */
    CUFFTDX_TRAIT_OUTPUT_TYPE = 2,
    /** Trait data type: long long int */
    CUFFTDX_TRAIT_IMPLICIT_TYPE_BATCHING = 3,
    /** Trait data type: long long int.
     * Expected content: >= 0 */
    CUFFTDX_TRAIT_ELEMENTS_PER_THREAD = 4,
    /** Trait data type: long long int.
     * Expected content: >= 0, in compute type elements */
    CUFFTDX_TRAIT_STORAGE_SIZE = 5,
    /** Trait data type: long long int.
     * Expected content: >= 0 */
    CUFFTDX_TRAIT_STRIDE = 6,
    /** Trait data type: long long int * 3.
     * Expected content: <x, y, z> block dimensions */
    CUFFTDX_TRAIT_BLOCK_DIM = 7,
    /** Trait data type: long long int (SMEM size in bytes) */
    CUFFTDX_TRAIT_SHARED_MEMORY_SIZE = 8,
    /** Trait data type: long long int */
    CUFFTDX_TRAIT_FFTS_PER_BLOCK = 9,
    /** Trait data type: char* */
    CUFFTDX_TRAIT_SYMBOL_NAME = 10,
    /** Trait data type: long long int, in input type elements */
    CUFFTDX_TRAIT_INPUT_LENGTH = 11,
    /** Trait data type: long long int, in output type elements */
    CUFFTDX_TRAIT_OUTPUT_LENGTH = 12,
    /** Trait data type: long long int */
    CUFFTDX_TRAIT_INPUT_ELEMENTS_PER_THREAD = 13,
    /** Trait data type: long long int */
    CUFFTDX_TRAIT_OUTPUT_ELEMENTS_PER_THREAD = 14,
    /** Trait data type: long long int */
    CUFFTDX_TRAIT_SUGGESTED_FFTS_PER_BLOCK = 15,
} cufftdxTraitType;

/**
 * @brief Creates a cuFFTDx descriptor
 * @param[in,out] handle A pointer to a cuFFTDx descriptor handle, and as output, a valid initialized descriptor.
 * @return `COMMONDX_SUCCESS` on success, or an error code
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxCreateDescriptor(cufftdxDescriptor* handle)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Set a C-string option on a cuFFTDx descriptor
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] opt The option to set
 * @param[out] value The C-string to set the option to
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxSetOptionStr(cufftdxDescriptor handle,
                                                                    commondxOption opt,
                                                                    const char* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns the number of knobs for a set of knobs
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] num_knobs The number of knobs
 * @param[in] knobs_ptr An array of num_knobs knobs
 * @param[out] size The number of distinct sets of knobs.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxGetKnobInt64Size(cufftdxDescriptor handle,
                                                                        size_t num_knobs,
                                                                        cufftdxKnobType* knobs_ptr,
                                                                        size_t* size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns the knobs values for a set of knobs
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] num_knobs The number of knobs
 * @param[in] knobs_ptr A pointer to an array of num_knobs knobs
 * @param[in] size The number of knobs.
 * @param[out] values The knob values. Must be a pointer to an array of at least size knobs values (integer)
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxGetKnobInt64s(cufftdxDescriptor handle,
                                                                     size_t num_knobs,
                                                                     cufftdxKnobType* knobs_ptr,
                                                                     size_t size,
                                                                     long long int* values) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Set an integer operator to a cuFFTDx descriptor
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] op The operator to set the descriptor to
 * @param[in] value The value to set the operator to
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxSetOperatorInt64(cufftdxDescriptor handle,
                                                                        cufftdxOperatorType op,
                                                                        long long int value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Set an array operator to a cuFFTDx descriptor
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] op The operator to set the descriptor to
 * @param[in] count The array size
 * @param[in] array A pointer to at least count integers, the arrat to set the descriptor to.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL
cufftdxSetOperatorInt64s(cufftdxDescriptor handle, cufftdxOperatorType op, size_t count, const long long int* array)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Get the LTOIR's size from a cuFFTDx descriptor
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[out] lto_size The size (in bytes) of the LTOIR
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxGetLTOIRSize(cufftdxDescriptor handle,
                                                                    size_t* lto_size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Get the LTOIR from a cuFFTDx descriptor
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] size The LTOIR size, in bytes
 * @param[out] lto The LTOIR code.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxGetLTOIR(cufftdxDescriptor handle,
                                                                size_t size,
                                                                void* lto) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns a C-string trait's value size
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] trait The trait to query the descriptor for
 * @param[out] size The C-string length (including `\0`)
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxGetTraitStrSize(cufftdxDescriptor handle,
                                                                       cufftdxTraitType trait,
                                                                       size_t* size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns a C-string trait value.
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] trait The trait to query the descriptor for
 * @param[in] size The C-string size (including the `\0`)
 * @param[out] value As output, the C-string trait value
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxGetTraitStr(cufftdxDescriptor handle,
                                                                   cufftdxTraitType trait,
                                                                   size_t size,
                                                                   char* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns an integer trait.
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] trait The trait to query the descriptor for
 * @param[out] value The trait integer value.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxGetTraitInt64(cufftdxDescriptor handle,
                                                                     cufftdxTraitType trait,
                                                                     long long int* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns an array of integers trait
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] trait The trait to query the descriptor for
 * @param[in] count The array size
 * @param[out] array The trait array.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxGetTraitInt64s(cufftdxDescriptor handle,
                                                                      cufftdxTraitType trait,
                                                                      size_t count,
                                                                      long long int* array) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Return a commondxValueType trait
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @param[in] trait The trait to query the descriptor for, of value commondxValueType
 * @param[out] value As output, the valuetype for the given input trait.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxGetTraitCommondxDataType(cufftdxDescriptor handle,
                                                                                cufftdxTraitType trait,
                                                                                commondxValueType* value)
    LIBMATHDX_API_NOEXCEPT;


/**
 * @brief Convert a cufftdxOperatorType instance to a human readable C-string
 *
 * @param[in] op A cufftdxOperatorType instance
 * @return A C-string
 */
LIBMATHDX_API const char* LIBMATHDX_CALL cufftdxOperatorTypeToStr(cufftdxOperatorType op) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Convert a cufftdxTraitType instance to a human readable C-string
 *
 * @param[in] op A cufftdxTraitType instance
 * @return A C-string
 */
LIBMATHDX_API const char* LIBMATHDX_CALL cufftdxTraitTypeToStr(cufftdxTraitType op) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Generate code from the cuFFTDx descriptor and stores it in code
 * @param[out] code A commondxCode instance, output of commondxCreateCode.
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxFinalizeCode(commondxCode code,
                                                                    cufftdxDescriptor handle) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Destroys a cuFFTDx descriptor
 * @param[in] handle A cuFFTDx descriptor, output of \ref cufftdxCreateDescriptor
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cufftdxDestroyDescriptor(cufftdxDescriptor handle)
    LIBMATHDX_API_NOEXCEPT;

#if defined(__cplusplus)
} // extern "C"
#endif /* __cplusplus */

#endif // MATHDX_CUFFTDX_H
