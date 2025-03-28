// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

// libmathdx's API is subject to change.
// Please contact Math-Libs-Feedback@nvidia.com for usage feedback.

#ifndef MATHDX_LIBCUBLASDX_H
#define MATHDX_LIBCUBLASDX_H

#include <stddef.h>

#include "libcommondx.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * @brief A handle to a cuBLASDx descriptor.
 *
 * Equivalent to `using GEMM = ...` in cuBLASDx CUDA C++.
 */
typedef long long int cublasdxDescriptor;

/**
 * @brief A handle to an opaque device tensor.
 */
typedef long long int cublasdxTensor;

/**
 * @brief A handle to a device function.
 * A device function operators on tensors described by \ref cublasdxTensor
 */
typedef long long int cublasdxDeviceFunction;

/**
 * @brief Type of cublasDx API
 *
 * Handling problems with default or custom/dynamic leading dimensions.
 * Check cublasdx::LeadingDimension operator documentation for more details
 * (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#leadingdimension-operator)
 */
typedef enum cublasdxApi_t {
    /**
     * Use API for problems with default leading dimensions.
     * Function API is defined by its signature:
     * `void (value_type_c* alpha, value_type_a* smem_a, value_type_b* smem_b, value_type_c* beta, value_type_c*
     * smem_c)` where
     *     - `smem_a`, `smem_b` and `smem_c` are pointers to value of type given by the \ref CUBLASDX_TRAIT_VALUE_TYPE
     * a, b, and c trait. `smem_a`, `smem_b` and `smem_c` must be shared memory pointers.
     *     - `alpha` and `beta` are pointers to value of type \ref CUBLASDX_TRAIT_VALUE_TYPE c.
     *
     * Note that complex numbers must be over-aligned.
     *
     * The function is `extern "C"` and the symbol name can be queried using \ref CUBLASDX_TRAIT_SYMBOL_NAME.
     * See `cuBLASDx documentation <https://docs.nvidia.com/cuda/cublasdx/api/methods.html#shared-memory-api>`_ and in
     * particular the `#2 - Pointer API` section.
     */
    CUBLASDX_API_SMEM = 0,
    /** Use API for problems with custom / dynamic leading dimensions.
     * Function API is defined by its signature:
     * `void (value_type_c alpha, value_type_a* smem_a, unsigned* lda, value_type_b *smem_b, unsigned* ldb,
     * value_type_c* beta, value_type_c* smem_c, unsigned *ldc)` where
     *     - `smem_a`, `smem_b` and `smem_c` are pointers to value of type given by the \ref CUBLASDX_TRAIT_VALUE_TYPE
     * a, b, and c trait. `smem_a`, `smem_b` and `smem_c` must be shared memory pointers.
     *     - `alpha` and `beta` are pointers to value of type \ref CUBLASDX_TRAIT_VALUE_TYPE c trait.
     *     - `lda`, `ldb` and `ldc` are pointers to unsigned 32 bits integer (`unsigned`)
     *
     * Note that complex numbers must be over-aligned.
     *
     * The function is `extern "C"` and the symbol name can be queried using \ref CUBLASDX_TRAIT_SYMBOL_NAME.
     * See `cuBLASDx documentation <https://docs.nvidia.com/cuda/cublasdx/api/methods.html#shared-memory-api>`_ and in
     * particular the `#3 - Pointer API, which allows providing runtime/dynamic leading dimensions for matrices A, B,
     * and C` section.
     */
    CUBLASDX_API_SMEM_DYNAMIC_LD = 1,
    /** Use Tensor API.
     * Function API is defined by the input and output tensors specified
     * when calling \ref cublasdxBindDeviceFunction
     * The device functions are not `extern "C"`. The name and argument types must be mangled.
     */
    CUBLASDX_API_TENSORS = 2,
} cublasdxApi;

/**
 * @brief Type of computation data
 *
 * Check cubladx::Type operator documentation for more details
 * (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#type-operator)
 */
typedef enum cublasdxType_t {
    /** Use for real matmuls */
    CUBLASDX_TYPE_REAL = 0,
    /** Use for complex matmuls */
    CUBLASDX_TYPE_COMPLEX = 1,
} cublasdxType;

/**
 * @brief Tensor transpose mode
 *
 * The transpose mode depends on cubladx::TransposeMode operator
 * which is deprecated since cublasDx 0.2.0 and might be removed in future
 * versions of mathDx libraries
 *
 * Check cubladx::TransposeMode operator documentation for more details
 * (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#transposemode-operator)
 */
typedef enum cublasdxTransposeMode_t {
    /** Use matrix as-is in the matmul */
    CUBLASDX_TRANSPOSE_MODE_NON_TRANSPOSED = 0,
    /** Use transposed matrix in the matmul */
    CUBLASDX_TRANSPOSE_MODE_TRANSPOSED = 1,
    /** Use transposed and conjugate matrix in the matmul */
    CUBLASDX_TRANSPOSE_MODE_CONJ_TRANSPOSED = 2,
} cublasdxTransposeMode;

/**
 * @brief Data arrangement mode
 *
 * Defines data arrangements in tensors' taking part in the calculation.
 *
 * Check cubladx::TransposeMode operator documentation for more details
 * (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#arrangement-operator)
 */
typedef enum cublasdxArrangement_t {
    /** Data is considered column-major */
    CUBLASDX_ARRANGEMENT_COL_MAJOR = 0,
    /** Data is considered row-major */
    CUBLASDX_ARRANGEMENT_ROW_MAJOR = 1,
} cublasdxArrangement;

/**
 * @brief BLAS function
 *
 * Sets the BLAS function that will be executed.
 *
 * Check cubladx::Function operator documentation for more details
 * (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#function-operator)
 */
typedef enum cublasdxFunction_t {
    /** Matrix-multiply */
    CUBLASDX_FUNCTION_MM = 0,
} cublasdxFunction;

/**
 * @brief cublasDx operators
 *
 * The set of supported cublasDx operators.
 *
 * Check cublaDx description operator documentation for more details
 * (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#function-operator)
 *
 * Check cublasDx execution operator documentation for more details
 * (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#execution-operators)
 */
typedef enum cublasdxOperatorType_t {
    /** Operator data type: \ref cublasdxFunction_t.
     * Operator definition: required */
    CUBLASDX_OPERATOR_FUNCTION = 0,
    /** Operator data type: long long int * 3.
     * Expected content: <M, N, K> problem sizes.
     * Operator definition: required */
    CUBLASDX_OPERATOR_SIZE = 1,
    /** Operator data type: \ref cublasdxType_t.
     * Operator definition: optional */
    CUBLASDX_OPERATOR_TYPE = 2,
    /** Operator data type: \ref commondxPrecision_t * 3.
     * Expected content: <A, B, C> precisions.
     * Operator definition: required */
    CUBLASDX_OPERATOR_PRECISION = 3,
    /** Operator data type: long long int.
     * Expected content: 700 (Volta), 800 (Ampere), ....
     * Operator definition: required */
    CUBLASDX_OPERATOR_SM = 4,
    /** Operator data type: \ref commondxExecution_t.
     * Operator definition: required */
    CUBLASDX_OPERATOR_EXECUTION = 5,
    /** Operator data type: long long int * 3.
     * Expected content: <x, y, z> block dimensions.
     * Operator definition: optional */
    CUBLASDX_OPERATOR_BLOCK_DIM = 6,
    /** Operator data type: long long int * 3.
     * Expected content: <LDA, LDB, LDC> leading dimensions.
     * Operator definition: optional */
    CUBLASDX_OPERATOR_LEADING_DIMENSION = 7,
    /** Operator data type: \ref cublasdxTransposeMode_t * 2.
     * Expected content: <A, B> transpose modes.
     * Operator definition: optional */
    CUBLASDX_OPERATOR_TRANSPOSE_MODE = 8,
    /** Operator data type: \ref cublasdxApi_t.
     * Operator definition: required */
    CUBLASDX_OPERATOR_API = 9,
    /** Operator data type: \ref cublasdxArrangement_t * 3.
     * Expected content: <A, B, C> data arrangements.
     * Operator definition: optional */
    CUBLASDX_OPERATOR_ARRANGEMENT = 10,
    /** Operator data type: long long int * 3.
     * Expected content: <AAlign, BAlign, CAlign> tensors' alignments.
     * Operator definition: optional */
    CUBLASDX_OPERATOR_ALIGNMENT = 11,
    /** Operator data type: long long int.
     * Expected content: 1, to enable cublasdx::experimental::StaticBlockDim.
     * Operator definition: optional */
    CUBLASDX_OPERATOR_STATIC_BLOCK_DIM = 12,
} cublasdxOperatorType;

/**
 * @brief cublasDx traits
 *
 * The set of supported types of traits that can be accessed from finalized sources
 * that use cublasDx.
 *
 * Check cublasDx Execution Block Traits documentation for more details
 * (https://docs.nvidia.com/cuda/cublasdx/api/traits.html#block-traits)
 */
typedef enum cublasdxTraitType_t {
    /** Trait data type: \ref commondxValueType_t * 3.
     * Expected content: <A, B, C> types.
     */
    CUBLASDX_TRAIT_VALUE_TYPE = 0,
    /** Trait data type: long long int * 3.
     * Expected content: <M, N, K> problem sizes.
     */
    CUBLASDX_TRAIT_SIZE = 1,
    /** Trait data type: long long int.
     * Expected content: multiplication result of block dimensions (x * y * z).
     */
    CUBLASDX_TRAIT_BLOCK_SIZE = 2,
    /** Trait data type: long long int * 3.
     * Expected content: <x, y, z> block dimension.
     */
    CUBLASDX_TRAIT_BLOCK_DIM = 3,
    /** Trait data type: long long int * 3.
     * Expected content: <LDA, LDB, LDC> leading dimensions.
     */
    CUBLASDX_TRAIT_LEADING_DIMENSION = 4,
    /** Trait data type: C-string
     */
    CUBLASDX_TRAIT_SYMBOL_NAME = 5,
    /** Trait data type: \ref cublasdxArrangement_t * 3.
     * Expected content: <A, B, C> data arrangements.
     */
    CUBLASDX_TRAIT_ARRANGEMENT = 6,
    /** Trait data type: long long int * 3.
     * Expected content: <AAlign, BAlign, CAlign> tensors' alignments, in bytes.
     */
    CUBLASDX_TRAIT_ALIGNMENT = 7,
    /** Trait data type: long long int * 3.
     * Expected content: <LDA, LDB, LDC>.
     */
    CUBLASDX_TRAIT_SUGGESTED_LEADING_DIMENSION = 8,
    /** Trait data type: long long int * 3.
     * Expected content: <X, Y, Z>.
     */
    CUBLASDX_TRAIT_SUGGESTED_BLOCK_DIM = 9,
    /** Trait data type: long long int.
     * Expected content: the product of three elements in block dimension.
     */
    CUBLASDX_TRAIT_MAX_THREADS_PER_BLOCK = 10
} cublasdxTraitType;

/**
 * @brief cuBLASDx desired tensor type
 *
 * Tensor types are opaque (layout is unspecified), non-owning, and defined by
 * - Memory space (global, shared or register memory)
 * - Size & alignment (in bytes)
 *
 * Tensor's representation in memory depends on their memory space.
 * Shared & register tensors are defined as
 *
 * \code
 * struct OpTensor {
 *   void* ptr;
 * }
 * \endcode
 *
 * Global memory tensors have an associated runtime leading dimension, and their
 * representation is
 *
 * \code
 * struct OpLdTensor {
 *   void* ptr;
 *   long long ld;
 * }
 * \endcode
 *
 * In either case, `ptr` must point to some storage (with appropriate size and alignment,
 * see below) and is not owning. The user is expected to keep memory allocated beyond
 * any use of the tensor. `ld` is a signed, 64bit integer equal to the leading dimension of the
 * global memory tensor. The leading dimension is the number of *elements* between two successive rows or
 * columns, depending on the context.
 *
 * All tensor APIs take their argument by value (not by pointer) and expect the struct to be passed
 * as-is on the stack.
 *
 * Each opaque tensor type is uniquely identified by a unique LD, see \ref cublasdxTensorTrait_t
 */
typedef enum cublasdxTensorType_t {
    /**
     * A shared memory tensor for `A`, in simple row or column layout
     * Corresponds to cuBLASDx `make_tensor(..., get_layout_smem_a());`
     **/
    CUBLASDX_TENSOR_SMEM_A = 0,
    /**
     * A shared memory tensor for `B`, in simple row or column layout.
     * Corresponds to cuBLASDx `make_tensor(..., get_layout_smem_b());`
     **/
    CUBLASDX_TENSOR_SMEM_B = 1,
    /**
     * A shared memory tensor for `C`, in simple row or column layout.
     * Corresponds to cuBLASDx `make_tensor(..., get_layout_smem_c());`
     **/
    CUBLASDX_TENSOR_SMEM_C = 2,
    /**
     * A shared memory tensor for `A`, in unspecified layout.
     * Corresponds to cuBLASDx `make_tensor(..., suggest_layout_smem_a());`
     **/
    CUBLASDX_TENSOR_SUGGESTED_SMEM_A = 3,
    /**
     * A shared memory tensor for `B`, in unspecified layout.
     * Corresponds to cuBLASDx `make_tensor(..., suggest_layout_smem_b());`
     **/
    CUBLASDX_TENSOR_SUGGESTED_SMEM_B = 4,
    /**
     * A shared memory tensor for `C`, in unspecified layout.
     * Corresponds to cuBLASDx `make_tensor(..., suggest_layout_smem_c());`
     **/
    CUBLASDX_TENSOR_SUGGESTED_SMEM_C = 5,
    /**
     * A register tensor for `C`, in unspecified layout.
     * Corresponds to cuBLASDx `suggest_partitioner().make_accumulator_fragment();`
     **/
    CUBLASDX_TENSOR_SUGGESTED_RMEM_C = 6,
    /**
     * A global memory view for `A` (typically a tile of a larger matrix)
     * in row or column major format, with a runtime leading dimension (`lda`).
     * Corresponds to cuBLASDx `make_tensor(a, get_layout_gmem_a(lda));`
     **/
    CUBLASDX_TENSOR_GMEM_A = 7,
    /**
     * A global memory view for `B` (typically a tile of a larger matrix)
     * in row or column major format, with a runtime leading dimension (`ldb`).
     * Corresponds to cuBLASDx `make_tensor(a, get_layout_gmem_b(ldb));`
     **/
    CUBLASDX_TENSOR_GMEM_B = 8,
    /**
     * A global memory view for `C` (typically a tile of a larger matrix)
     * in row or column major format, with a runtime leading dimension (`ldc`).
     * Corresponds to cuBLASDx `make_tensor(a, get_layout_gmem_c(ldc));`
     **/
    CUBLASDX_TENSOR_GMEM_C = 9,
} cublasdxTensorType;

/**
 * @brief Tensor traits, used to query informations
 *
 */
typedef enum cublasdxTensorTrait_t {
    /**
     * The size of the underlying storage, in bytes.
     * Trait data type: long long int.
     */
    CUBLASDX_TENSOR_TRAIT_STORAGE_BYTES = 0,
    /**
     * The alignment of the underlying storage, in bytes.
     * Trait data type: long long int.
     */
    CUBLASDX_TENSOR_TRAIT_ALIGNMENT_BYTES = 1,
    /**
     * The tensor type UID. Tensor types with the same UID
     * are identical and can be passed through various cuBLASDx
     * device functions.
     * Trait data type: long long int.
     */
    CUBLASDX_TENSOR_TRAIT_UID = 2,
    /**
     * A human readable C-string representing the opaque tensor template
     * name, either `cuBLASDxOpTensor` or `cuBLASDxOpLdTensor`.
     * Trait data type: C-string.
     */
    CUBLASDX_TENSOR_TRAIT_NAME = 3,
    /**
     * A human readable C-string representing the opaque tensor instantiation,
     * either `cuBLASDxOpTensor<..>` or `cuBLASDxOpLdTensor<...>`.
     * Trait data type: C-string.
     */
    CUBLASDX_TENSOR_TRAIT_OPAQUE_NAME = 4,
} cublasdxTensorTrait;

/**
 * @brief Device function traits, used to query informations
 */
typedef enum cublasdxDeviceFunctionTrait_t {
    /**
     * The name of the device function.
     * All device functions are templated on their
     * input type. This means the signature is
     * template<typename T0, typename T1, typename T2, ...>
     * void <name>(T0, T1, T2, ...)
     * and must be mangled in order to be called as-is.
     * Trait data type: C-string
     */
    CUBLASDX_DEVICE_FUNCTION_TRAIT_NAME = 0,
    /**
     * The mangled name (aka symbol) of the device function.
     * Trait data type: C-string
     */
    CUBLASDX_DEVICE_FUNCTION_TRAIT_SYMBOL = 1,
} cublasdxDeviceFunctionTrait;

/**
 * @brief Device function traits, used to query informations
 */
typedef enum cublasdxDeviceFunctionOption_t {
    /**
     * Specify an optional alignment option for copy functions, in bytes.
     * Must be a power of 2 between 1 and 16.
     * Trait data type: long long int
     */
    CUBLASDX_DEVICE_FUNCTION_OPTION_COPY_ALIGNMENT = 0,
} cublasdxDeviceFunctionOption;

/**
 * @brief Device functions supported by the library
 */
typedef enum cublasdxDeviceFunctionType_t {
    /**
     * Execute the device function (matmul).
     * When C is a register tensor, the device function API is
     * `execute(A, B, C)` which computes `C += A x B`.
     * When C is a shared memory tensor, the device function API is
     * `execute(alpha, A, B, beta, C)` which computes
     * `C = alpha A x B + beta C`. A, B and C are tensors, while
     * alpha and beta are scalars of type value_type_c.
     *
     * Inputs:
     *      - A, an instance of `CUBLASDX_TENSOR_SUGGESTED_SMEM_A` or
     *                          `CUBLASDX_TENSOR_SMEM_A`
     *      - B, an instance of `CUBLASDX_TENSOR_SUGGESTED_SMEM_B`
     *                          `CUBLASDX_TENSOR_SMEM_B`
     *      - C, an instance of `CUBLASDX_TENSOR_SUGGESTED_SMEM_C` or
     *                          `CUBLASDX_TENSOR_SMEM_C` or
     *                          `CUBLASDX_TENSOR_SUGGESTED_RMEM_C`
     *
     * Outputs:
     *      - C, an instance of `CUBLASDX_TENSOR_SUGGESTED_SMEM_C` or
     *                          `CUBLASDX_TENSOR_SMEM_C` or
     *                          `CUBLASDX_TENSOR_SUGGESTED_RMEM_C`
     */
    CUBLASDX_DEVICE_FUNCTION_EXECUTE = 0,
    /**
     * Copies from one tensor to another. `copy(S, D)` copies
     * from `S` to `D`.
     *
     * Inputs & outputs: two distinct tensors, instances of
     *      - `CUBLASDX_TENSOR_SUGGESTED_SMEM_A`
     *      - `CUBLASDX_TENSOR_SUGGESTED_SMEM_B`
     *      - `CUBLASDX_TENSOR_SUGGESTED_SMEM_C`
     *      - `CUBLASDX_TENSOR_SUGGESTED_RMEM_C`
     *      - `CUBLASDX_TENSOR_GMEM_A`
     *      - `CUBLASDX_TENSOR_GMEM_B`
     *      - `CUBLASDX_TENSOR_GMEM_C`
     *
     * Tensors can be in different memory spaces but must correspond to the
     * same A, B or C matrix.
     */
    CUBLASDX_DEVICE_FUNCTION_COPY = 1,
    /**
     * Wait on all previously issued copies to complete.
     * `wait_all()` waits on all previously issued copies to complete.
     * Inputs & outputs: none
     */
    CUBLASDX_DEVICE_FUNCTION_COPY_WAIT = 2,
    /**
     * Zeroes out a tensor. `clear(C)` zeroes out `C`.
     * Inputs & outputs:
     *     - An instance of `CUBLASDX_TENSOR_SUGGESTED_RMEM_C`
     */
    CUBLASDX_DEVICE_FUNCTION_CLEAR = 3,
} cublasdxDeviceFunctionType;

/**
 * @brief Creates a cuBLASDx descriptor
 *
 * @param[out] handle A pointer to a handle
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxCreateDescriptor(cublasdxDescriptor* handle)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Sets an option on a cuBLASDx descriptor
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] option An option to set the descriptor to.
 * @param[in] value A pointer to a C-string
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxSetOptionStr(cublasdxDescriptor handle,
                                                                     commondxOption option,
                                                                     const char* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Set an operator on a cuBLASDx descriptor to an integer value
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] op An operator to set the descriptor to.
 * @param[in] value The operator's value
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxSetOperatorInt64(cublasdxDescriptor handle,
                                                                         cublasdxOperatorType op,
                                                                         long long int value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Set an operator on a cuBLASDx descriptor to an integer array
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] op An option to set the descriptor to.
 * @param[in] count The size of the operator array, as indicated by the \ref cublasdxOperatorType_t documentation
 * @param[in] array A pointer to an array containing the operator's value. Must point to at least `count` elements.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL
cublasdxSetOperatorInt64s(cublasdxDescriptor handle, cublasdxOperatorType op, size_t count, const long long int* array)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Binds (aka create) a tensor handle to a tensor
 * The tensor is bound to the cuBLASDx descriptor and will be freed when the cuBLASDx descriptor is
 * destroyed.
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] tensor_type The tensor type to bind to the handle
 * @param[out] tensor A valid tensor handle
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxBindTensor(cublasdxDescriptor handle,
                                                                   cublasdxTensorType tensor_type,
                                                                   cublasdxTensor* tensor) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Finalize the tensors. This is required before traits can be queried.
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] count The number of tensors to finalized
 * @param[out] array The array of tensors
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL
cublasdxFinalizeTensors(cublasdxDescriptor handle, size_t count, const cublasdxTensor* array) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Query an integer trait value from a finalized tensor
 *
 * @param[in] tensor A finalized tensor handle, output of \ref cublasdxBindTensor
 * @param[in] trait The trait to query
 * @param[out] value The trait value
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetTensorTraitInt64(cublasdxTensor tensor,
                                                                            cublasdxTensorTrait trait,
                                                                            long long int* value)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Query an C-string trait's size from a finalized tensor
 *
 * @param[in] tensor A finalized tensor handle, output of \ref cublasdxBindTensor
 * @param[in] trait The trait to query
 * @param[out] size The C-string size (including the `\0`)
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetTensorTraitStrSize(cublasdxTensor tensor,
                                                                              cublasdxTensorTrait trait,
                                                                              size_t* size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Query a C-string trait value from a finalized tensor
 *
 * @param[in] tensor A finalized tensor handle, output of \ref cublasdxBindTensor
 * @param[in] trait The trait to query
 * @param[in] size The C-string size, as returned by \ref cublasdxGetTensorTraitStrSize
 * @param[out] value The C-string trait value
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetTensorTraitStr(cublasdxTensor tensor,
                                                                          cublasdxTensorTrait trait,
                                                                          size_t size,
                                                                          char* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Binds (aka create) a device function from a set of tensor
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] device_function_type The device function to create
 * @param[in] count The number of input & output tensors to the device function
 * @param[in] array The array of input & output tensors
 * @param[out] device_function The device function
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL
cublasdxBindDeviceFunction(cublasdxDescriptor handle,
                           cublasdxDeviceFunctionType device_function_type,
                           size_t count,
                           const cublasdxTensor* array,
                           cublasdxDeviceFunction* device_function) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Finalize (aka codegen) a set of device function into a code handle
 *
 * After this, LTOIR can be extracted from `code` using \ref commondxGetCodeLTOIR
 *
 * @param[out] code A code handle, output from \ref commondxCreateCode
 * @param[in] count The number of device functions to codegen
 * @param[in] array The array of device functions
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxFinalizeDeviceFunctions(commondxCode code,
                                                                                size_t count,
                                                                                const cublasdxDeviceFunction* array)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Query a device function C-string trait value size
 *
 * @param[in] device_function A device function handle, output from \ref cublasdxFinalizeDeviceFunctions
 * @param[in] trait The trait to query the device function
 * @param[out] size The size of the trait value C-string, including the `\0`
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL
cublasdxGetDeviceFunctionTraitStrSize(cublasdxDeviceFunction device_function,
                                      cublasdxDeviceFunctionTrait trait,
                                      size_t* size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Query a device function C-string trait value
 *
 * @param[in] device_function A device function handle, output from \ref cublasdxFinalizeDeviceFunctions
 * @param[in] trait The trait to query the device function
 * @param[in] size The size of the trait value C-string as returned by \ref cublasdxGetDeviceFunctionTraitStrSize
 * @param[out] value The trait value as a C-string. Must point to at least `size` bytes.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL
cublasdxGetDeviceFunctionTraitStr(cublasdxDeviceFunction device_function,
                                  cublasdxDeviceFunctionTrait trait,
                                  size_t size,
                                  char* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns the LTOIR size, in bytes
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[out] lto_size The size of the LTOIR.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetLTOIRSize(cublasdxDescriptor handle,
                                                                     size_t* lto_size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns the LTOIR
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] size The size, in bytes, of the LTOIR, as returned by \ref cublasdxGetLTOIRSize
 * @param[out] lto A pointer to at least `size` bytes containing the LTOIR
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetLTOIR(cublasdxDescriptor handle,
                                                                 size_t size,
                                                                 void* lto) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns the size of a C-string trait
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] trait The trait to query the size of
 * @param[out] size The size of the C-string value, including the `\0`.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetTraitStrSize(cublasdxDescriptor handle,
                                                                        cublasdxTraitType trait,
                                                                        size_t* size) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns a C-string trait's value
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] trait The trait to query on the descriptor
 * @param[in] size The size of the C-string (including the `\0`)
 * @param[out] value The C-string trait value. Must point to at least `size` bytes.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetTraitStr(cublasdxDescriptor handle,
                                                                    cublasdxTraitType trait,
                                                                    size_t size,
                                                                    char* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns an integer trait's value
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] trait A trait to query the handle for
 * @param[out] value The trait value.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetTraitInt64(cublasdxDescriptor handle,
                                                                      cublasdxTraitType trait,
                                                                      long long int* value) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns an array trait's value
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] trait A trait to query handle for
 * @param[in] count The number of elements in the trait array, as indicated in the \ref cublasdxTraitType_t
 * documentation.
 * @param[out] array A pointer to at least count integers. As output, an array filled with the trait value.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetTraitInt64s(cublasdxDescriptor handle,
                                                                       cublasdxTraitType trait,
                                                                       size_t count,
                                                                       long long int* array) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns an array trait's value, when the elements are commondxValueType values
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @param[in] trait A trait to query handle for
 * @param[in] count The number of elements in the trait array, as indicated in the \ref cublasdxTraitType_t
 * documentation.
 * @param[out] array A pointer to at least count commondxValueType. As output, an array filled with the trait value.
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxGetTraitCommondxDataTypes(cublasdxDescriptor handle,
                                                                                  cublasdxTraitType trait,
                                                                                  size_t count,
                                                                                  commondxValueType* array)
    LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Convert an operator enum to a human readable C-string
 *
 * @param[in] op The operator enum to convert
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API const char* LIBMATHDX_CALL cublasdxOperatorTypeToStr(cublasdxOperatorType op) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Convert a trait enum to a human readable C-string
 *
 * @param[in] trait The trait enum to convert
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API const char* LIBMATHDX_CALL cublasdxTraitTypeToStr(cublasdxTraitType trait) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Fill an instance of commondxCode with the code from the cuBLASDx descriptor
 *
 * @param[out] code A commondxCode code
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxFinalizeCode(commondxCode code,
                                                                     cublasdxDescriptor handle) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Destroy a cuBLASDx descriptor
 *
 * @param[in] handle A cuBLASDx descriptor, output of \ref cublasdxCreateDescriptor
 * @return `COMMONDX_SUCCESS` on success, or an error code.
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL cublasdxDestroyDescriptor(cublasdxDescriptor handle)
    LIBMATHDX_API_NOEXCEPT;

#if defined(__cplusplus)
} // extern "C"
#endif /* __cplusplus */

#endif // MATHDX_LIBCUBLASDX_H
