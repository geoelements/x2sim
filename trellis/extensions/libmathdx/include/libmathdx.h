// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

// libmathdx's API is subject to change.
// Please contact Math-Libs-Feedback@nvidia.com for usage feedback.

#ifndef MATHDX_MATHDX_H
#define MATHDX_MATHDX_H

#include "libcommondx.h"
#include "libcublasdx.h"
#include "libcufftdx.h"
#include "libcusolverdx.h"

#define LIBMATHDX_VER_MAJOR 0
#define LIBMATHDX_VER_MINOR 2
#define LIBMATHDX_VER_PATCH 0

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Returns the libmathdx version as a single integer
 *
 * @param[out] version The version, encoded as 1000 * major + 100 * minor + patch
 * @return `COMMONDX_SUCCESS`
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL mathdxGetVersion(int* version) LIBMATHDX_API_NOEXCEPT;

/**
 * @brief Returns the libmathdx version as a triplet of integers
 *
 * @param[out] major The major version.
 * @param[out] minor The minor version.
 * @param[out] patch The patch version.
 * @return `COMMONDX_SUCCESS`
 */
LIBMATHDX_API commondxStatusType LIBMATHDX_CALL mathdxGetVersionEx(int* major,
                                                                   int* minor,
                                                                   int* patch) LIBMATHDX_API_NOEXCEPT;

#if defined(__cplusplus)
} // extern "C"
#endif /* __cplusplus */

#endif // MATHDX_MATHDX_H
