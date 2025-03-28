// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

// libmathdx's API is subject to change.
// Please contact Math-Libs-Feedback@nvidia.com for usage feedback.

#ifndef MATHDX_CUSOLVER_H
#define MATHDX_CUSOLVER_H

#include <stddef.h>

#include "libcommondx.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * @brief A cuSOLVERDx descriptor
 *
 * Equivalent to `using SOLVER = ...` in cuSOLVERDx C++.
 */
typedef long long int cusolverdxDescriptor;

/**
 * @brief Type of cusolverdx API
 */
typedef enum cusolverdxApi_t {
    /**
     * Input-output is in shared memory.
     * Function API is defined by its signature and \ref cusolverdxFunction_t.
     *
     * For \ref CUSOLVERDX_FUNCTION_POTRF: `void (value_type* A, status_type* info)`
     *     - `A` is a pointer to a shared memory array of value_type values.
     *     - `status_type` is a pointer to a 32b integer (`int`).
     *
     * For \ref CUSOLVERDX_FUNCTION_POTRS: `void (value_type* A, value_type* B)`
     *     - `A` and `B` are pointer to shared memory arrays of value_type values.
     *
     * Those functions are `extern "C"` and the symbol name can be queried using \ref CUSOLVERDX_TRAIT_SYMBOL_NAME
     */
    CUSOLVERDX_API_SMEM = 0,
    /**
     * Input-output is in shared memory.
     * Function API is defined by its signature and \ref cusolverdxFunction_t.
     *
     * For \ref CUSOLVERDX_FUNCTION_POTRF: `void (value_type* A, unsigned* lda, status_type* info)`
     *     - `A` is a pointer to a shared memory array of value_type values.
     *     - `lda` is a pointer to a unsigned 32b integer (`unsigned`).
     *     - `status_type` is a pointer to a signed 32b integer (`int`).
     *
     * For \ref CUSOLVERDX_FUNCTION_POTRS: `void (value_type* A, unsigned* lda, value_type* B, unsigned* ldb)`
     *     - `A` and `B` are pointer to shared memory arrays of value_type values.
     *     - `lda` and `ldb` are pointers to unsigned 32b integers (`unsigned`), representing the leading dimensions of
     * A and B.
     *
     * Those functions are `extern "C"` and the symbol name can be queried using \ref CUSOLVERDX_TRAIT_SYMBOL_NAME.
     */
    CUSOLVERDX_API_SMEM_DYNAMIC_LD = 1,
} cusolverdxApi;

/**
 * @brief Type of input values
 */
typedef enum cusolverdxType_t {
    /** Input and output is real */
    CUSOLVERDX_TYPE_REAL = 0,
} cusolverdxType;

/**
 * @brief Type of device function
 */
typedef enum cusolverdxFunction_t {
    /**
     * Cholesky factorization
     * See cuSOLVERDx Cholesky factorization for more details
     * (https://docs.nvidia.com/cuda/cusolverdx/get_started/potrf.html)
     */
    CUSOLVERDX_FUNCTION_POTRF = 2,
    /**
     * Cholesky solve
     * See cuSOLVERDx Cholesky solve for more details
     * (https://docs.nvidia.com/cuda/cusolverdx/get_started/potrs.html)
     */
    CUSOLVERDX_FUNCTION_POTRS = 3,
} cusolverdxFunction;

/**
 * @brief Data arrangement mode
 *
 * Defines data arrangements in tensors' taking part in the calculation.
 */
typedef enum cusolverdxArrangement_t {
    /** Input and output are column major */
    CUSOLVERDX_ARRANGEMENT_COL_MAJOR = 0,
    /** Input and output are row major */
    CUSOLVERDX_ARRANGEMENT_ROW_MAJOR = 1,
} cusolverdxArrangement;

/**
 * @brief Tensor fill mode
 *
 * For symmetric matrix the fill mode can be upper or lower triangular
 */
typedef enum cusolverdxFillMode_t {
    /** Upper-triangular */
    CUSOLVERDX_FILL_MODE_UPPER = 0,
    /** Lower-triangular */
    CUSOLVERDX_FILL_MODE_LOWER = 1,
} cusolverdxFillMode;

/**
 * @brief Operators
 *
 * The set of supported cusolverDx operators.
 */
typedef enum cusolverdxOperatorType_t {
    /** Operator data type: long long int * 2.
     * Expected content: <M, N> problem sizes.
     * Operator definition: required */
    CUSOLVERDX_OPERATOR_SIZE = 0,
    /** Operator data type: \ref cusolverdxType_t.
     * Operator definition: required */
    CUSOLVERDX_OPERATOR_TYPE = 1,
    /** Operator data type: \ref commondxPrecision_t.
     * Operator definition: required */
    CUSOLVERDX_OPERATOR_PRECISION = 2,
    /** Operator data type: long long int.
     * Expected content: 700 (Volta), 800 (Ampere), ...,
     * Operator definition: required */
    CUSOLVERDX_OPERATOR_SM = 3,
    /** Operator data type: \ref commondxExecution_t.
     * Operator definition: required */
    CUSOLVERDX_OPERATOR_EXECUTION = 4,
    /** Operator data type: long long * 3.
     * Expected content: <x, y, z> block dimensions.
     * Operator definition: optional */
    CUSOLVERDX_OPERATOR_BLOCK_DIM = 5,
    /** Operator data type: \ref cusolverdxApi_t.
     * Operator definition: required */
    CUSOLVERDX_OPERATOR_API = 6,
    /** Operator data type: \ref cusolverdxFunction_t.
     * Operator definition: required */
    CUSOLVERDX_OPERATOR_FUNCTION = 7,
    /** Operator data type: \ref cusolverdxArrangement_t.
     * Operator definition: optional */
    CUSOLVERDX_OPERATOR_ARRANGEMENT = 8,
    /** Operator data type: \ref cusolverdxFillMode_t.
     * Operator definition: optional */
    CUSOLVERDX_OPERATOR_FILL_MODE = 9
} cusolverdxOperatorType;

/**
 * @brief Traits
 *
 * The set of supported types of traits that can be accessed from finalized sources
 * that use cusolverdx.
 */
typedef enum cusolverdxTraitType_t {
    /** Trait data type: long long int.
     * Value: shared memory size, in bytes. */
    CUSOLVERDX_TRAIT_SHARED_MEMORY_SIZE = 1,
    /** Trait data type: C-string
     * Value: symbol (device function) name. */
    CUSOLVERDX_TRAIT_SYMBOL_NAME = 2,
} cusolverdxTraitType;

/**
 * @brief Creates a cuSOLVERDx descriptor
 * @param[in,out] handle A pointer to a descriptor handle. As output, an initialized cuSOLVERDx descriptor
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxCreateDescriptor(cusolverdxDescriptor* handle)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Sets a C-string option on a cuSOLVERDx descriptor
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[in] opt The option to set
 * @param[in] value The value for the option
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxSetOptionStr(cusolverdxDescriptor handle,
                                                                       commondxOption opt,
                                                                       const char* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Sets an integer operator on a cuSOLVERDx descriptor
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[in] op The operator to set.
 * @param[in] value A value for the operator
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxSetOperatorInt64(cusolverdxDescriptor handle,
                                                                           cusolverdxOperatorType op,
                                                                           long long int value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Sets a integer array operator on a cuSOLVERDx descriptor
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[in] op The operator to set
 * @param[in] count The number of entries in the array value, as indicated in the \ref cusolverdxOperatorType_t
 * documentation.
 * @param[in] array A pointer to at least count integers, the array operator to set
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxSetOperatorInt64s(cusolverdxDescriptor handle,
                                                                            cusolverdxOperatorType op,
                                                                            size_t count,
                                                                            const long long int* array)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Extract the size of the LTOIR for a cuSOLVERDx descriptor
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[out] lto_size As output, the size of the LTOIR
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxGetLTOIRSize(cusolverdxDescriptor handle,
                                                                       size_t* lto_size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Extract the LTOIR from a cuSOLVERDx descriptor
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[in] size The LTOIR size, output of \ref cusolverdxGetLTOIRSize
 * @param[out] lto The buffer contains the LTOIR
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxGetLTOIR(cusolverdxDescriptor handle,
                                                                   size_t size,
                                                                   void* lto) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns the size of the universal fatbin for cuSOLVERDx
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[out] fatbin_size The size of the fatbin, in bytes
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL
cusolverdxGetUniversalFATBINSize(cusolverdxDescriptor handle, size_t* fatbin_size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns a universal fatbin for cuSOLVERDx
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[in] fatbin_size The size of the fatbin, output from \ref cusolverdxGetUniversalFATBINSize
 * @param[out] fatbin The universal fatbin. Must pointer to at least `fatbin_size` bytes.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxGetUniversalFATBIN(cusolverdxDescriptor handle,
                                                                             size_t fatbin_size,
                                                                             void* fatbin) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns the size of a C-string trait value
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[in] trait A trait to query the descriptor for
 * @param[out] size The size of the C-string value for the trait (including the `\0`)
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxGetTraitStrSize(cusolverdxDescriptor handle,
                                                                          cusolverdxTraitType trait,
                                                                          size_t* size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns a C-string trait value
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[in] trait A trait to query the descriptor for
 * @param[in] size The size of the C-string, output from \ref cusolverdxGetTraitStrSize
 * @param[out] value The C-string trait value
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxGetTraitStr(cusolverdxDescriptor handle,
                                                                      cusolverdxTraitType trait,
                                                                      size_t size,
                                                                      char* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns an integer trait value
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @param[in] trait A trait to query the descriptor for
 * @param[out] value The trait value
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxGetTraitInt64(cusolverdxDescriptor handle,
                                                                        cusolverdxTraitType trait,
                                                                        long long int* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Converts an operator enum to a human readable C-string
 * @param[in] op An operator enum
 * @return A human readable C-string
 */
LIBMATHDX_API const char* LIBMATHDX_CALL cusolverdxOperatorTypeToStr(cusolverdxOperatorType op) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Converts a trait enum to a human readable C-string
 * @param[in] op A trait enum
 * @return A human readable C-string
 */
LIBMATHDX_API const char* LIBMATHDX_CALL cusolverdxTraitTypeToStr(cusolverdxTraitType op) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Fills a code handle with the descriptor's device function code
 * @param[out] code A code handle
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxFinalizeCode(commondxCode code, cusolverdxDescriptor handle)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Destroys a cuSOLVERDx descriptor
 * @param[in] handle A cuSOLVERDx descriptor, output of \ref cusolverdxCreateDescriptor
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cusolverdxDestroyDescriptor(cusolverdxDescriptor handle)
    LIBMATHDX_API_NOEXCEPT;

#if defined(__cplusplus)
} // extern "C"
#endif /* __cplusplus */

#endif // MATHDX_CUSOLVER_H
