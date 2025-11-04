/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cudf {
namespace strings {
namespace detail {

// Helper function for loading 16B from a potentially unaligned memory location to registers.
__forceinline__ __device__ uint4 load_uint4(char const* ptr)
{
    auto const offset       = reinterpret_cast<std::uintptr_t>(ptr) % 4;
    auto const* aligned_ptr = reinterpret_cast<unsigned int const*>(ptr - offset);
    auto const shift        = offset * 8;

    uint4 regs = {aligned_ptr[0], aligned_ptr[1], aligned_ptr[2], aligned_ptr[3]};
    uint tail  = 0;
    if (shift) tail = aligned_ptr[4];

    regs.x = __funnelshift_r(regs.x, regs.y, shift);
    regs.y = __funnelshift_r(regs.y, regs.z, shift);
    regs.z = __funnelshift_r(regs.z, regs.w, shift);
    regs.w = __funnelshift_r(regs.w, tail, shift);

    return regs;
}

/**
* @brief Returns a new strings column using the specified indices to select
* elements from the `strings` column.
*
* Caller must update the validity mask in the output column.
*
* ```
* s1 = ["a", "b", "c", "d", "e", "f"]
* map = [0, 2]
* s2 = gather<true>( s1, map.begin(), map.end() )
* s2 is ["a", "c"]
* ```
*
* @tparam NullifyOutOfBounds If true, indices outside the column's range are nullified.
* @tparam MapIterator Iterator for retrieving integer indices of the column.
*
* @param strings Strings instance for this operation.
* @param begin Start of index iterator.
* @param end End of index iterator.
* @param stream CUDA stream used for device memory operations and kernel launches.
* @param mr Device memory resource used to allocate the returned column's device memory.
* @return New strings column containing the gathered strings.
*/
template <bool NullifyOutOfBounds, typename MapIterator>
std::unique_ptr<column> gather(strings_column_view const& strings,
                               MapIterator begin,
                               MapIterator end,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

/**
* @brief Returns a new strings column using the specified indices to select
* elements from the `strings` column.
*
* Caller must update the validity mask in the output column.
*
* ```
* s1 = ["a", "b", "c", "d", "e", "f"]
* map = [0, 2]
* s2 = gather( s1, map.begin(), map.end(), true )
* s2 is ["a", "c"]
* ```
*
* @tparam MapIterator Iterator for retrieving integer indices of the column.
*
* @param strings Strings instance for this operation.
* @param begin Start of index iterator.
* @param end End of index iterator.
* @param nullify_out_of_bounds If true, indices outside the column's range are nullified.
* @param stream CUDA stream used for device memory operations and kernel launches.
* @param mr Device memory resource used to allocate the returned column's device memory.
* @return New strings column containing the gathered strings.
*/                               
template <typename MapIterator>
std::unique_ptr<column> gather(strings_column_view const& strings,
                               MapIterator begin,
                               MapIterator end,
                               bool nullify_out_of_bounds,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf


