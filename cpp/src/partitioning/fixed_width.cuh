/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace cudf::detail {

enum class fixed_width_metadata_layout { packed32, unpacked32, generic };

using fixed_width_partition_result = std::pair<std::unique_ptr<table>, std::vector<size_type>>;

inline int fixed_width_required_bits(std::uint64_t count) noexcept
{
  int bits  = 0;
  auto last = count > 0 ? count - 1 : 0;
  while (last != 0) {
    last >>= 1;
    ++bits;
  }
  return bits;
}

inline fixed_width_metadata_layout select_fixed_width_metadata_layout(
  size_type num_partitions,
  size_type packed_rows_per_thread,
  size_type unpacked_rows_per_thread) noexcept
{
  auto const partition_bits = fixed_width_required_bits(static_cast<std::uint64_t>(num_partitions));
  auto const packed_fits =
    packed_rows_per_thread > 0 &&
    partition_bits + fixed_width_required_bits(1024ULL * packed_rows_per_thread) <= 32;
  if (packed_fits) { return fixed_width_metadata_layout::packed32; }
  return unpacked_rows_per_thread > 0 ? fixed_width_metadata_layout::unpacked32
                                      : fixed_width_metadata_layout::generic;
}

/**
 * @brief Returns true when every key column can use the fixed-width partition path.
 */
inline bool is_fixed_width_partition_compatible(table_view const& keys)
{
  return std::all_of(keys.begin(), keys.end(), [](column_view const& column) {
    return cudf::is_fixed_width(column.type());
  });
}

/**
 * @brief Attempts the fixed-width hash-partition implementation.
 *
 * Returns `std::nullopt` when the selected device cannot fit a legal optimized
 * launch. Unsupported key types must be filtered with
 * `is_fixed_width_partition_compatible` before calling this function.
 */
std::optional<fixed_width_partition_result> try_fixed_width_hash_partition(
  table_view const& input,
  table_view const& keys,
  size_type num_partitions,
  hash_id hash_function,
  uint32_t seed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
