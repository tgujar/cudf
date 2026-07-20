/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fixed_width.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cuda/pipeline>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/scan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace cudf::detail {
namespace {

namespace cg = cooperative_groups;

constexpr size_type block_size             = fixed_width_partition_block_size;
constexpr std::size_t async_copy_alignment = 16;
static_assert(block_size % cudf::detail::warp_size == 0);

enum class metadata_kind { packed32, unpacked32 };

struct fixed_width_launch_config {
  size_type rows_per_thread;
  size_type grid_size;
};

template <cudf::type_id Id>
struct dispatch_fixed_width_type {
  using dispatched_type = id_to_type<Id>;
  using type =
    cuda::std::conditional_t<cudf::is_fixed_width<dispatched_type>(), dispatched_type, void>;
};

template <typename HashValue>
class bitwise_partitioner {
 public:
  CUDF_HOST_DEVICE explicit bitwise_partitioner(size_type count) : _mask{count - 1} {}
  __device__ size_type operator()(HashValue value) const { return value & _mask; }

 private:
  size_type _mask;
};

template <typename HashValue>
class modulo_partitioner {
 public:
  CUDF_HOST_DEVICE explicit modulo_partitioner(size_type count) : _divisor{count} {}
  __device__ size_type operator()(HashValue value) const { return value % _divisor; }

 private:
  size_type _divisor;
};

CUDF_HOST_DEVICE constexpr std::size_t round_up(std::size_t value, std::size_t alignment)
{
  return (value + alignment - 1) / alignment * alignment;
}

template <typename Key>
struct identity_hash {
  using result_type = uint32_t;

  CUDF_HOST_DEVICE constexpr explicit identity_hash(uint32_t = 0) {}

  template <typename Return = result_type>
  __device__ constexpr Return operator()(Key const& key) const
    requires(cuda::std::is_arithmetic_v<Key>)
  {
    return static_cast<Return>(key);
  }

  template <typename Return = result_type>
  __device__ constexpr Return operator()(Key const&) const
    requires(!cuda::std::is_arithmetic_v<Key>)
  {
    CUDF_UNREACHABLE("Identity hash requires numeric keys");
  }
};

template <metadata_kind Kind>
struct metadata_view;

template <>
struct metadata_view<metadata_kind::packed32> {
  std::uint32_t* values;
  int partition_bits;

  __device__ void store(size_type index, size_type partition, size_type offset) const
  {
    values[index] = (static_cast<std::uint32_t>(offset) << partition_bits) |
                    static_cast<std::uint32_t>(partition);
  }

  __device__ size_type partition(size_type index) const
  {
    auto const mask =
      partition_bits == 0 ? std::uint32_t{0} : (std::uint32_t{1} << partition_bits) - 1;
    return static_cast<size_type>(values[index] & mask);
  }

  __device__ size_type offset(size_type index) const
  {
    return static_cast<size_type>(values[index] >> partition_bits);
  }
};

template <>
struct metadata_view<metadata_kind::unpacked32> {
  size_type* partitions;
  size_type* offsets;

  __device__ void store(size_type index, size_type partition, size_type offset) const
  {
    partitions[index] = partition;
    offsets[index]    = offset;
  }

  __device__ size_type partition(size_type index) const { return partitions[index]; }
  __device__ size_type offset(size_type index) const { return offsets[index]; }
};

struct fixed_width_column_descriptor {
  std::uint8_t const* input;
  std::uint8_t* output;
  size_type width;
};

__device__ size_type local_row_index(size_type iteration)
{
  return iteration * static_cast<size_type>(blockDim.x) + static_cast<size_type>(threadIdx.x);
}

__device__ thread_index_type global_row_index(size_type iteration)
{
  auto const first =
    static_cast<thread_index_type>(blockIdx.x) * static_cast<thread_index_type>(blockDim.x) +
    static_cast<thread_index_type>(threadIdx.x);
  auto const stride =
    static_cast<thread_index_type>(gridDim.x) * static_cast<thread_index_type>(blockDim.x);
  return first + static_cast<thread_index_type>(iteration) * stride;
}

template <template <typename> class Hash>
struct direct_element_hasher {
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  __device__ hash_value_type operator()(column_device_view const& column,
                                        size_type row,
                                        uint32_t seed) const
  {
    return Hash<T>{seed}(column.element<T>(row));
  }

  template <typename T, CUDF_ENABLE_IF(not cudf::is_fixed_width<T>())>
  __device__ hash_value_type operator()(column_device_view const&, size_type, uint32_t) const
  {
    CUDF_UNREACHABLE("Unsupported key type in fixed-width partition");
  }
};

template <template <typename> class Hash, bool HasNulls>
__device__ hash_value_type hash_key_row(table_device_view const& keys, size_type row, uint32_t seed)
{
  direct_element_hasher<Hash> hasher;
  auto const null_hash = cuda::std::numeric_limits<hash_value_type>::max();
  auto const& first    = keys.column(0);
  auto hash            = HasNulls && first.is_null(row) ? null_hash
                                                        : cudf::type_dispatcher<dispatch_fixed_width_type>(
                                                 first.type(), hasher, first, row, seed);

  for (size_type column_index = 1; column_index < keys.num_columns(); ++column_index) {
    auto const& column      = keys.column(column_index);
    auto const element_hash = HasNulls && column.is_null(row)
                                ? null_hash
                                : cudf::type_dispatcher<dispatch_fixed_width_type>(
                                    column.type(), hasher, column, row, seed);
    hash                    = cudf::hashing::detail::hash_combine(hash, element_hash);
  }
  return hash;
}

template <template <typename> class Hash, bool HasNulls, typename Partitioner, metadata_kind Kind>
CUDF_KERNEL void materialize_metadata_kernel(table_device_view keys,
                                             size_type num_rows,
                                             size_type num_partitions,
                                             size_type rows_per_thread,
                                             uint32_t seed,
                                             Partitioner partitioner,
                                             metadata_view<Kind> metadata,
                                             size_type* block_partition_sizes)
{
  extern __shared__ size_type histogram[];

  for (size_type partition = threadIdx.x; partition < num_partitions; partition += blockDim.x) {
    histogram[partition] = 0;
  }
  __syncthreads();

  for (size_type iteration = 0; iteration < rows_per_thread; ++iteration) {
    auto const global_row = global_row_index(iteration);
    if (global_row < static_cast<thread_index_type>(num_rows)) {
      auto const row       = static_cast<size_type>(global_row);
      auto const partition = partitioner(hash_key_row<Hash, HasNulls>(keys, row, seed));
      auto const offset    = atomicAdd(histogram + partition, size_type{1});
      metadata.store(row, partition, offset);
    }
  }
  __syncthreads();

  for (size_type partition = threadIdx.x; partition < num_partitions; partition += blockDim.x) {
    block_partition_sizes[static_cast<std::size_t>(partition) * gridDim.x + blockIdx.x] =
      histogram[partition];
  }
}

template <typename Word>
__device__ void full_warp_memcpy_async(cg::thread_block_tile<cudf::detail::warp_size> const& warp,
                                       Word* destination,
                                       Word const* source,
                                       cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
  constexpr auto bytes_to_copy = cudf::detail::warp_size * sizeof(Word);
  static_assert(bytes_to_copy % async_copy_alignment == 0);

  pipe.producer_acquire();
  cuda::memcpy_async(
    warp, destination, source, cuda::aligned_size_t<async_copy_alignment>(bytes_to_copy), pipe);
  pipe.producer_commit();
}

template <typename Word>
__device__ void partial_warp_memcpy_async(
  cg::thread_block_tile<cudf::detail::warp_size> const& warp,
  Word* destination,
  Word const* source,
  size_type words_to_copy,
  cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
  auto const lane = static_cast<size_type>(warp.thread_rank());
  if (lane < words_to_copy) {
    pipe.producer_acquire();
    cuda::memcpy_async(
      destination + lane, source + lane, cuda::aligned_size_t<alignof(Word)>(sizeof(Word)), pipe);
    pipe.producer_commit();
  }
}

template <int BlockSize>
__device__ void scan_partition_counts(size_type const* block_partition_sizes,
                                      size_type* local_offsets,
                                      size_type num_partitions)
{
  using block_scan = cub::BlockScan<size_type, BlockSize>;
  __shared__ typename block_scan::TempStorage scan_storage;
  __shared__ size_type carry;
  __shared__ size_type tile_base_offset;

  if (threadIdx.x == 0) { carry = 0; }
  __syncthreads();

  for (size_type tile = 0; tile < num_partitions; tile += BlockSize) {
    auto const partition = tile + static_cast<size_type>(threadIdx.x);
    auto const value =
      partition < num_partitions
        ? block_partition_sizes[static_cast<std::size_t>(partition) * gridDim.x + blockIdx.x]
        : size_type{0};
    size_type prefix{};
    size_type aggregate{};
    block_scan(scan_storage).ExclusiveSum(value, prefix, aggregate);
    __syncthreads();

    if (threadIdx.x == 0) {
      tile_base_offset = carry;
      carry += aggregate;
    }
    __syncthreads();

    if (partition < num_partitions) { local_offsets[partition] = tile_base_offset + prefix; }
    __syncthreads();
  }

  if (threadIdx.x == 0) { local_offsets[num_partitions] = carry; }
  __syncthreads();
}

template <std::size_t Size>
struct alignas(Size) byte_chunk {
  std::uint8_t data[Size];
};

template <std::size_t Size>
__device__ void copy_aligned_chunks(std::uint8_t*& destination,
                                    std::uint8_t const*& source,
                                    size_type& bytes)
{
  auto const addresses =
    reinterpret_cast<std::uintptr_t>(destination) | reinterpret_cast<std::uintptr_t>(source);
  if ((addresses & (Size - 1)) != 0) { return; }
  while (bytes >= static_cast<size_type>(Size)) {
    *reinterpret_cast<byte_chunk<Size>*>(destination) =
      *reinterpret_cast<byte_chunk<Size> const*>(source);
    destination += Size;
    source += Size;
    bytes -= Size;
  }
}

__device__ void copy_bytes(std::uint8_t* destination, std::uint8_t const* source, size_type bytes)
{
  copy_aligned_chunks<16>(destination, source, bytes);
  copy_aligned_chunks<8>(destination, source, bytes);
  copy_aligned_chunks<4>(destination, source, bytes);
  copy_aligned_chunks<2>(destination, source, bytes);
  while (bytes-- > 0) {
    *destination++ = *source++;
  }
}

template <typename Word>
__device__ void copy_typed_column(fixed_width_column_descriptor descriptor,
                                  std::uint8_t* payload,
                                  std::uint32_t const* local_slots,
                                  size_type num_rows,
                                  size_type num_partitions,
                                  size_type rows_per_thread,
                                  size_type const* local_partition_offsets,
                                  size_type const* global_partition_offsets,
                                  size_type flush_tile_size)
{
  auto const* input  = reinterpret_cast<Word const*>(descriptor.input);
  auto* output       = reinterpret_cast<Word*>(descriptor.output);
  auto* block_output = reinterpret_cast<Word*>(payload);

  for (size_type iteration = 0; iteration < rows_per_thread; ++iteration) {
    auto const global_row = global_row_index(iteration);
    if (global_row >= static_cast<thread_index_type>(num_rows)) { continue; }
    auto const row = static_cast<size_type>(global_row);

    auto const slot    = local_slots[local_row_index(iteration)];
    block_output[slot] = input[row];
  }
  __syncthreads();

  auto const tile =
    cg::tiled_partition(cg::this_thread_block(), static_cast<unsigned int>(flush_tile_size));
  auto const tiles_per_block = static_cast<size_type>(blockDim.x / flush_tile_size);
  auto const tile_index      = static_cast<size_type>(threadIdx.x / flush_tile_size);

  for (size_type partition = tile_index; partition < num_partitions; partition += tiles_per_block) {
    auto const partition_size =
      local_partition_offsets[partition + 1] - local_partition_offsets[partition];
    for (size_type offset = static_cast<size_type>(tile.thread_rank()); offset < partition_size;
         offset += static_cast<size_type>(tile.size())) {
      auto const source_slot = local_partition_offsets[partition] + offset;
      auto const output_slot = global_partition_offsets[partition] + offset;
      output[output_slot]    = block_output[source_slot];
    }
  }
  __syncthreads();
}

__device__ void copy_generic_column(fixed_width_column_descriptor descriptor,
                                    std::uint8_t* payload,
                                    std::uint32_t const* local_slots,
                                    size_type num_rows,
                                    size_type num_partitions,
                                    size_type rows_per_thread,
                                    size_type const* local_partition_offsets,
                                    size_type const* global_partition_offsets,
                                    size_type flush_tile_size)
{
  auto pipe = cuda::make_pipeline();
  for (size_type iteration = 0; iteration < rows_per_thread; ++iteration) {
    auto const global_row = global_row_index(iteration);
    if (global_row >= static_cast<thread_index_type>(num_rows)) { continue; }
    auto const row = static_cast<size_type>(global_row);

    auto const slot = local_slots[local_row_index(iteration)];
    pipe.producer_acquire();
    cuda::memcpy_async(payload + static_cast<std::size_t>(slot) * descriptor.width,
                       descriptor.input + static_cast<std::size_t>(row) * descriptor.width,
                       static_cast<std::size_t>(descriptor.width),
                       pipe);
    pipe.producer_commit();
  }
  cuda::pipeline_consumer_wait_prior<0>(pipe);
  pipe.consumer_release();
  __syncthreads();

  auto const tile =
    cg::tiled_partition(cg::this_thread_block(), static_cast<unsigned int>(flush_tile_size));
  auto const tiles_per_block = static_cast<size_type>(blockDim.x / flush_tile_size);
  auto const tile_index      = static_cast<size_type>(threadIdx.x / flush_tile_size);

  for (size_type partition = tile_index; partition < num_partitions; partition += tiles_per_block) {
    auto const partition_size =
      local_partition_offsets[partition + 1] - local_partition_offsets[partition];
    for (size_type offset = static_cast<size_type>(tile.thread_rank()); offset < partition_size;
         offset += static_cast<size_type>(tile.size())) {
      auto const source_slot = local_partition_offsets[partition] + offset;
      auto const output_slot = global_partition_offsets[partition] + offset;
      copy_bytes(descriptor.output + static_cast<std::size_t>(output_slot) * descriptor.width,
                 payload + static_cast<std::size_t>(source_slot) * descriptor.width,
                 descriptor.width);
    }
  }
  __syncthreads();
}

template <metadata_kind Kind>
CUDF_KERNEL void fused_fixed_width_copy_kernel(fixed_width_column_descriptor const* descriptors,
                                               size_type num_columns,
                                               size_type num_rows,
                                               size_type num_partitions,
                                               size_type rows_per_thread,
                                               size_type max_column_width,
                                               metadata_view<Kind> global_metadata,
                                               size_type const* block_partition_sizes,
                                               size_type const* scanned_block_partition_sizes,
                                               size_type flush_tile_size)
{
  extern __shared__ __align__(16) std::uint8_t shared[];
  auto const rows_per_block     = static_cast<std::size_t>(blockDim.x) * rows_per_thread;
  auto const payload_bytes      = round_up(rows_per_block * max_column_width, async_copy_alignment);
  auto* payload                 = shared;
  auto* local_slots             = reinterpret_cast<std::uint32_t*>(shared + payload_bytes);
  auto* local_partition_offsets = reinterpret_cast<size_type*>(local_slots + rows_per_block);
  auto* global_partition_offsets = local_partition_offsets + num_partitions + 1;

  auto const warp    = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto metadata_pipe = cuda::make_pipeline();
  if constexpr (Kind == metadata_kind::packed32) {
    auto const warp_base = static_cast<thread_index_type>(warp.meta_group_rank()) *
                           static_cast<thread_index_type>(warp.size());
    auto const first =
      static_cast<thread_index_type>(blockIdx.x) * static_cast<thread_index_type>(blockDim.x) +
      warp_base;
    auto const stride =
      static_cast<thread_index_type>(gridDim.x) * static_cast<thread_index_type>(blockDim.x);
    for (size_type iteration = 0; iteration < rows_per_thread; ++iteration) {
      auto const local_base =
        iteration * static_cast<size_type>(blockDim.x) + static_cast<size_type>(warp_base);
      auto const row_base = first + static_cast<thread_index_type>(iteration) * stride;
      if (row_base < static_cast<thread_index_type>(num_rows)) {
        auto const rows_remaining = static_cast<thread_index_type>(num_rows) - row_base;
        auto const rows_to_copy   = rows_remaining < cudf::detail::warp_size
                                      ? static_cast<size_type>(rows_remaining)
                                      : static_cast<size_type>(cudf::detail::warp_size);
        if (rows_to_copy == cudf::detail::warp_size) {
          full_warp_memcpy_async(
            warp, local_slots + local_base, global_metadata.values + row_base, metadata_pipe);
        } else {
          partial_warp_memcpy_async(warp,
                                    local_slots + local_base,
                                    global_metadata.values + row_base,
                                    rows_to_copy,
                                    metadata_pipe);
        }
      }
    }
  }

  scan_partition_counts<block_size>(block_partition_sizes, local_partition_offsets, num_partitions);

  for (size_type partition = threadIdx.x; partition < num_partitions; partition += blockDim.x) {
    global_partition_offsets[partition] =
      scanned_block_partition_sizes[static_cast<std::size_t>(partition) * gridDim.x + blockIdx.x];
  }

  if constexpr (Kind == metadata_kind::packed32) {
    cuda::pipeline_consumer_wait_prior<0>(metadata_pipe);
    metadata_pipe.consumer_release();
    warp.sync();
  }

  if constexpr (Kind == metadata_kind::packed32) {
    auto const local_packed_metadata =
      metadata_view<metadata_kind::packed32>{local_slots, global_metadata.partition_bits};
    for (size_type iteration = 0; iteration < rows_per_thread; ++iteration) {
      auto const global_row = global_row_index(iteration);
      if (global_row >= static_cast<thread_index_type>(num_rows)) { continue; }

      auto const local     = local_row_index(iteration);
      auto const partition = local_packed_metadata.partition(local);
      auto const offset    = local_packed_metadata.offset(local);
      local_slots[local] = static_cast<std::uint32_t>(local_partition_offsets[partition] + offset);
    }
  } else {
    for (size_type iteration = 0; iteration < rows_per_thread; ++iteration) {
      auto const global_row = global_row_index(iteration);
      if (global_row >= static_cast<thread_index_type>(num_rows)) { continue; }
      auto const row = static_cast<size_type>(global_row);

      auto const local     = local_row_index(iteration);
      auto const partition = global_metadata.partition(row);
      auto const offset    = global_metadata.offset(row);
      local_slots[local] = static_cast<std::uint32_t>(local_partition_offsets[partition] + offset);
    }
  }
  __syncthreads();

  for (size_type column_index = 0; column_index < num_columns; ++column_index) {
    auto const descriptor = descriptors[column_index];
    switch (descriptor.width) {
      case 1:
        copy_typed_column<std::uint8_t>(descriptor,
                                        payload,
                                        local_slots,
                                        num_rows,
                                        num_partitions,
                                        rows_per_thread,
                                        local_partition_offsets,
                                        global_partition_offsets,
                                        flush_tile_size);
        break;
      case 2:
        copy_typed_column<std::uint16_t>(descriptor,
                                         payload,
                                         local_slots,
                                         num_rows,
                                         num_partitions,
                                         rows_per_thread,
                                         local_partition_offsets,
                                         global_partition_offsets,
                                         flush_tile_size);
        break;
      case 4:
        copy_typed_column<std::uint32_t>(descriptor,
                                         payload,
                                         local_slots,
                                         num_rows,
                                         num_partitions,
                                         rows_per_thread,
                                         local_partition_offsets,
                                         global_partition_offsets,
                                         flush_tile_size);
        break;
      case 8:
        copy_typed_column<std::uint64_t>(descriptor,
                                         payload,
                                         local_slots,
                                         num_rows,
                                         num_partitions,
                                         rows_per_thread,
                                         local_partition_offsets,
                                         global_partition_offsets,
                                         flush_tile_size);
        break;
      case 16:
        copy_typed_column<byte_chunk<16>>(descriptor,
                                          payload,
                                          local_slots,
                                          num_rows,
                                          num_partitions,
                                          rows_per_thread,
                                          local_partition_offsets,
                                          global_partition_offsets,
                                          flush_tile_size);
        break;
      default:
        copy_generic_column(descriptor,
                            payload,
                            local_slots,
                            num_rows,
                            num_partitions,
                            rows_per_thread,
                            local_partition_offsets,
                            global_partition_offsets,
                            flush_tile_size);
        break;
    }
  }
}

template <metadata_kind Kind>
CUDF_KERNEL void build_gather_map_kernel(size_type num_rows,
                                         size_type rows_per_thread,
                                         metadata_view<Kind> metadata,
                                         size_type const* scanned_block_partition_sizes,
                                         size_type* gather_map)
{
  for (size_type iteration = 0; iteration < rows_per_thread; ++iteration) {
    auto const global_row = global_row_index(iteration);
    if (global_row < static_cast<thread_index_type>(num_rows)) {
      auto const row       = static_cast<size_type>(global_row);
      auto const partition = metadata.partition(row);
      auto const output =
        scanned_block_partition_sizes[static_cast<std::size_t>(partition) * gridDim.x +
                                      blockIdx.x] +
        metadata.offset(row);
      gather_map[output] = row;
    }
  }
}

template <typename Kernel>
std::size_t dynamic_shared_memory_budget(Kernel kernel)
{
  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDF_CUDA_TRY(cudaGetDeviceProperties(&properties, device));

  cudaFuncAttributes attributes{};
  CUDF_CUDA_TRY(cudaFuncGetAttributes(&attributes, reinterpret_cast<void const*>(kernel)));
  auto const opt_in               = static_cast<std::size_t>(properties.sharedMemPerBlockOptin);
  auto const statically_allocated = static_cast<std::size_t>(attributes.sharedSizeBytes);
  if (attributes.maxThreadsPerBlock < block_size) { return 0; }
  return opt_in > statically_allocated ? opt_in - statically_allocated : 0;
}

template <typename Kernel>
int active_blocks_per_multiprocessor(Kernel kernel, std::size_t dynamic_shared_memory)
{
  auto const budget = dynamic_shared_memory_budget(kernel);
  if (dynamic_shared_memory > budget) { return 0; }
  auto const attribute_status = cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel),
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     static_cast<int>(budget));
  if (attribute_status != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }

  int active_blocks{};
  auto const occupancy_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, kernel, block_size, dynamic_shared_memory);
  if (occupancy_status != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return active_blocks;
}

template <typename Kernel>
void configure_dynamic_shared_memory(Kernel kernel)
{
  auto const budget = dynamic_shared_memory_budget(kernel);
  CUDF_CUDA_TRY(cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel),
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(budget)));
}

std::size_t histogram_shared_memory_size(size_type num_partitions)
{
  return static_cast<std::size_t>(num_partitions) * sizeof(size_type);
}

std::size_t copy_shared_memory_size(size_type num_partitions,
                                    size_type rows_per_thread,
                                    std::size_t max_fixed_width_size)
{
  auto const rows     = static_cast<std::size_t>(block_size) * rows_per_thread;
  auto const payload  = round_up(rows * max_fixed_width_size, async_copy_alignment);
  auto const metadata = rows * sizeof(std::uint32_t);
  auto const offsets  = (2 * static_cast<std::size_t>(num_partitions) + 1) * sizeof(size_type);
  return payload + metadata + offsets;
}

struct fixed_width_copy_batch {
  size_type column_offset;
  size_type column_count;
  size_type max_column_width;
  std::size_t shared_memory_size;
};

// Limit each batch to the estimated L2 working set of the concurrently resident CTAs. Keeping the
// input and partitioned output sectors in L2 reduces write amplification for wide tables.
template <metadata_kind Kind, typename Descriptors>
std::vector<fixed_width_copy_batch> make_copy_batches(Descriptors const& descriptors,
                                                      size_type grid_size,
                                                      size_type rows_per_thread,
                                                      size_type num_partitions)
{
  if (descriptors.empty()) { return {}; }

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDF_CUDA_TRY(cudaGetDeviceProperties(&properties, device));
  auto const l2_capacity = static_cast<std::uint64_t>(std::max(properties.l2CacheSize, 0));

  constexpr std::uint64_t sector_size = 32;
  auto const rows_per_block =
    static_cast<std::uint64_t>(block_size) * static_cast<std::uint64_t>(rows_per_thread);
  auto const partition_runs =
    std::min<std::uint64_t>(static_cast<std::uint64_t>(num_partitions), rows_per_block);
  constexpr std::uint64_t metadata_arrays =
    Kind == metadata_kind::packed32 ? std::uint64_t{1} : std::uint64_t{2};
  auto const copy_kernel = &fused_fixed_width_copy_kernel<Kind>;

  auto batch_footprint = [&](std::uint64_t column_footprint, size_type max_column_width) {
    auto const shared_memory =
      copy_shared_memory_size(num_partitions, rows_per_thread, max_column_width);
    auto const blocks_per_sm = active_blocks_per_multiprocessor(copy_kernel, shared_memory);
    if (blocks_per_sm <= 0) { return std::numeric_limits<std::uint64_t>::max(); }

    auto const resident_blocks =
      std::min<std::uint64_t>(static_cast<std::uint64_t>(grid_size),
                              static_cast<std::uint64_t>(blocks_per_sm) *
                                static_cast<std::uint64_t>(properties.multiProcessorCount));

    // Metadata is contiguous. Block counts and global partition offsets can each touch one sector
    // per partition and resident CTA.
    auto const routing_footprint =
      metadata_arrays * (rows_per_block * sizeof(std::uint32_t) +
                         static_cast<std::uint64_t>(rows_per_thread) * (sector_size - 1)) +
      2 * static_cast<std::uint64_t>(num_partitions) * sector_size;
    return resident_blocks * (routing_footprint + column_footprint);
  };

  std::vector<fixed_width_copy_batch> batches;
  auto column_offset = size_type{0};
  while (column_offset < static_cast<size_type>(descriptors.size())) {
    auto column_count     = size_type{0};
    auto max_column_width = size_type{0};
    auto column_footprint = std::uint64_t{0};
    while (column_offset + column_count < static_cast<size_type>(descriptors.size())) {
      auto const width =
        static_cast<std::uint64_t>(descriptors[column_offset + column_count].width);
      auto const input_footprint =
        rows_per_block * width + static_cast<std::uint64_t>(rows_per_thread) * (sector_size - 1);
      auto const output_footprint = rows_per_block * width + partition_runs * (sector_size - 1);
      auto const candidate_column_footprint = column_footprint + input_footprint + output_footprint;
      auto const candidate_max_width = std::max(max_column_width, static_cast<size_type>(width));
      auto const candidate_footprint =
        batch_footprint(candidate_column_footprint, candidate_max_width);
      if (column_count > 0 && candidate_footprint > l2_capacity) { break; }
      ++column_count;
      max_column_width = candidate_max_width;
      column_footprint = candidate_column_footprint;
    }

    batches.push_back({column_offset,
                       column_count,
                       max_column_width,
                       copy_shared_memory_size(num_partitions, rows_per_thread, max_column_width)});
    column_offset += column_count;
  }
  return batches;
}

template <template <typename> class Hash, bool HasNulls, typename Partitioner, metadata_kind Kind>
std::optional<fixed_width_launch_config> compute_launch_config(size_type num_rows,
                                                               size_type num_partitions,
                                                               std::size_t max_fixed_width_size,
                                                               bool has_fixed_width_payload)
{
  auto const metadata_kernel = &materialize_metadata_kernel<Hash, HasNulls, Partitioner, Kind>;
  auto const histogram_bytes = histogram_shared_memory_size(num_partitions);
  if (active_blocks_per_multiprocessor(metadata_kernel, histogram_bytes) <= 0) {
    return std::nullopt;
  }

  auto const make_launch_config = [num_rows](size_type rows_per_thread) {
    auto const rows_per_block =
      static_cast<std::uint64_t>(block_size) * static_cast<std::uint64_t>(rows_per_thread);
    auto const grid_size =
      cudf::util::div_rounding_up_safe(static_cast<std::uint64_t>(num_rows), rows_per_block);
    return fixed_width_launch_config{rows_per_thread, static_cast<size_type>(grid_size)};
  };

  // With no fused payload copy, hashing has no per-row shared-memory requirement. Match the
  // original optimized partitioner schedule.
  constexpr size_type direct_hash_rows_per_thread = 8;
  if (!has_fixed_width_payload) { return make_launch_config(direct_hash_rows_per_thread); }

  auto const copy_kernel               = &fused_fixed_width_copy_kernel<Kind>;
  auto const copy_shared_memory_budget = dynamic_shared_memory_budget(copy_kernel);
  auto const fixed_copy_shared_memory_bytes =
    copy_shared_memory_size(num_partitions, 0, max_fixed_width_size);
  if (fixed_copy_shared_memory_bytes >= copy_shared_memory_budget) { return std::nullopt; }

  auto const copy_shared_memory_per_iteration =
    static_cast<std::size_t>(block_size) * (max_fixed_width_size + sizeof(std::uint32_t));
  auto candidate_rows_per_thread =
    static_cast<size_type>((copy_shared_memory_budget - fixed_copy_shared_memory_bytes) /
                           copy_shared_memory_per_iteration);
  while (candidate_rows_per_thread > 0) {
    auto const candidate_copy_shared_memory_bytes =
      copy_shared_memory_size(num_partitions, candidate_rows_per_thread, max_fixed_width_size);
    if (candidate_copy_shared_memory_bytes <= copy_shared_memory_budget &&
        active_blocks_per_multiprocessor(copy_kernel, candidate_copy_shared_memory_bytes) > 0) {
      break;
    }
    --candidate_rows_per_thread;
  }

  if (candidate_rows_per_thread <= 0) { return std::nullopt; }
  return make_launch_config(candidate_rows_per_thread);
}

template <template <typename> class Hash, bool HasNulls, typename Partitioner, metadata_kind Kind>
fixed_width_partition_result execute_fixed_width_partition(
  table_view const& input,
  table_view const& keys,
  size_type num_partitions,
  uint32_t seed,
  Partitioner partitioner,
  fixed_width_launch_config launch_config,
  std::vector<size_type> const& fixed_width_indices,
  std::vector<size_type> const& non_fixed_width_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{Kind == metadata_kind::packed32 ? "hash_partition_fixed_width_packed"
                                                           : "hash_partition_fixed_width_unpacked"};

  auto const num_rows              = input.num_rows();
  auto const rows_per_thread       = launch_config.rows_per_thread;
  auto const grid_size             = launch_config.grid_size;
  auto const metadata_count        = static_cast<std::size_t>(num_rows);
  auto const block_partition_count = static_cast<std::size_t>(grid_size) * num_partitions;
  auto const current_mr            = cudf::get_current_device_resource_ref();
  auto const partition_bits = fixed_width_required_bits(static_cast<std::uint64_t>(num_partitions));

  rmm::device_uvector<uint32_t> packed_metadata(
    Kind == metadata_kind::packed32 ? metadata_count : 0, stream, current_mr);
  rmm::device_uvector<size_type> unpacked_partitions(
    Kind == metadata_kind::unpacked32 ? metadata_count : 0, stream, current_mr);
  rmm::device_uvector<size_type> unpacked_offsets(
    Kind == metadata_kind::unpacked32 ? metadata_count : 0, stream, current_mr);
  auto metadata = [&] {
    if constexpr (Kind == metadata_kind::packed32) {
      return metadata_view<Kind>{packed_metadata.data(), partition_bits};
    } else {
      return metadata_view<Kind>{unpacked_partitions.data(), unpacked_offsets.data()};
    }
  }();

  rmm::device_uvector<size_type> block_partition_sizes(block_partition_count, stream, current_mr);
  rmm::device_uvector<size_type> scanned_block_partition_sizes(
    block_partition_count, stream, current_mr);

  auto key_device_view = table_device_view::create(keys, stream);

  cudf::detail::host_vector<size_type> host_partition_offsets(
    static_cast<std::size_t>(num_partitions) + 1,
    cudf::detail::rmm_host_allocator<size_type>{cudf::get_pinned_memory_resource(), stream});

  std::vector<std::unique_ptr<column>> fixed_width_outputs;
  fixed_width_outputs.reserve(fixed_width_indices.size());
  cudf::detail::host_vector<fixed_width_column_descriptor> host_column_descriptors(
    {cudf::get_pinned_memory_resource(), stream});
  host_column_descriptors.reserve(fixed_width_indices.size());

  auto const metadata_shared_memory_bytes = histogram_shared_memory_size(num_partitions);
  auto const metadata_kernel = &materialize_metadata_kernel<Hash, HasNulls, Partitioner, Kind>;
  configure_dynamic_shared_memory(metadata_kernel);
  metadata_kernel<<<grid_size, block_size, metadata_shared_memory_bytes, stream.value()>>>(
    *key_device_view,
    num_rows,
    num_partitions,
    rows_per_thread,
    seed,
    partitioner,
    metadata,
    block_partition_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, current_mr),
                         block_partition_sizes.begin(),
                         block_partition_sizes.end(),
                         scanned_block_partition_sizes.begin());

  CUDF_CUDA_TRY(cudaMemcpy2DAsync(host_partition_offsets.data(),
                                  sizeof(size_type),
                                  scanned_block_partition_sizes.data(),
                                  static_cast<std::size_t>(grid_size) * sizeof(size_type),
                                  sizeof(size_type),
                                  num_partitions,
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));

  for (auto index : fixed_width_indices) {
    auto const& source = input.column(index);
    auto output        = cudf::make_fixed_width_column(
      source.type(), source.size(), mask_state::UNALLOCATED, stream, mr);
    auto output_view = output->mutable_view();
    auto const width = static_cast<size_type>(cudf::size_of(source.type()));
    host_column_descriptors.push_back(
      fixed_width_column_descriptor{static_cast<uint8_t const*>(source.head()) +
                                      static_cast<std::size_t>(source.offset()) * width,
                                    static_cast<uint8_t*>(output_view.head()),
                                    width});
    fixed_width_outputs.push_back(std::move(output));
  }

  if (!fixed_width_indices.empty()) {
    rmm::device_uvector<fixed_width_column_descriptor> column_descriptors(
      host_column_descriptors.size(), stream, current_mr);
    cudf::detail::cuda_memcpy_async<fixed_width_column_descriptor>(
      column_descriptors, host_column_descriptors, stream);

    constexpr size_type flush_tile_size = cudf::detail::warp_size;

    auto const copy_kernel = &fused_fixed_width_copy_kernel<Kind>;
    configure_dynamic_shared_memory(copy_kernel);
    auto const batches =
      make_copy_batches<Kind>(host_column_descriptors, grid_size, rows_per_thread, num_partitions);
    for (auto const& batch : batches) {
      copy_kernel<<<grid_size, block_size, batch.shared_memory_size, stream.value()>>>(
        column_descriptors.data() + batch.column_offset,
        batch.column_count,
        num_rows,
        num_partitions,
        rows_per_thread,
        batch.max_column_width,
        metadata,
        block_partition_sizes.data(),
        scanned_block_partition_sizes.data(),
        flush_tile_size);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  }

  auto const fixed_width_input       = input.select(fixed_width_indices);
  auto const fixed_width_is_nullable = nullable(fixed_width_input);
  auto const needs_gather_map        = !non_fixed_width_indices.empty() || fixed_width_is_nullable;
  std::optional<rmm::device_uvector<size_type>> gather_map;
  if (needs_gather_map) {
    gather_map.emplace(num_rows, stream, current_mr);
    build_gather_map_kernel<Kind>
      <<<grid_size, block_size, 0, stream.value()>>>(num_rows,
                                                     rows_per_thread,
                                                     metadata,
                                                     scanned_block_partition_sizes.data(),
                                                     gather_map->data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  if (fixed_width_is_nullable) {
    detail::gather_bitmask(fixed_width_input,
                           gather_map->begin(),
                           fixed_width_outputs,
                           detail::gather_bitmask_op::DONT_CHECK,
                           stream,
                           mr);
  }

  std::vector<std::unique_ptr<column>> non_fixed_width_outputs;
  if (!non_fixed_width_indices.empty()) {
    auto gathered =
      detail::gather(input.select(non_fixed_width_indices),
                     device_span<size_type const>{gather_map->data(), gather_map->size()},
                     out_of_bounds_policy::DONT_CHECK,
                     negative_index_policy::NOT_ALLOWED,
                     stream,
                     mr);
    non_fixed_width_outputs = gathered->release();
  }

  std::vector<std::unique_ptr<column>> outputs(input.num_columns());
  for (std::size_t i = 0; i < fixed_width_indices.size(); ++i) {
    outputs[fixed_width_indices[i]] = std::move(fixed_width_outputs[i]);
  }
  for (std::size_t i = 0; i < non_fixed_width_indices.size(); ++i) {
    outputs[non_fixed_width_indices[i]] = std::move(non_fixed_width_outputs[i]);
  }

  stream.synchronize();
  host_partition_offsets[num_partitions] = num_rows;
  auto partition_offsets =
    std::vector<size_type>(host_partition_offsets.begin(), host_partition_offsets.end());
  return {std::make_unique<table>(std::move(outputs)), std::move(partition_offsets)};
}

template <template <typename> class Hash, bool HasNulls, typename Partitioner>
std::optional<fixed_width_partition_result> try_partitioner(table_view const& input,
                                                            table_view const& keys,
                                                            size_type num_partitions,
                                                            uint32_t seed,
                                                            Partitioner partitioner,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  std::vector<size_type> fixed_width_indices;
  std::vector<size_type> non_fixed_width_indices;
  std::size_t max_fixed_width_size = 0;
  for (size_type index = 0; index < input.num_columns(); ++index) {
    auto const& column = input.column(index);
    if (cudf::is_fixed_width(column.type())) {
      fixed_width_indices.push_back(index);
      max_fixed_width_size =
        std::max(max_fixed_width_size, static_cast<std::size_t>(cudf::size_of(column.type())));
    } else {
      non_fixed_width_indices.push_back(index);
    }
  }

  auto const num_rows                = input.num_rows();
  auto const has_fixed_width_payload = !fixed_width_indices.empty();
  auto const packed_config =
    compute_launch_config<Hash, HasNulls, Partitioner, metadata_kind::packed32>(
      num_rows, num_partitions, max_fixed_width_size, has_fixed_width_payload);
  if (packed_config && packed_metadata_fits(num_partitions, packed_config->rows_per_thread)) {
    return execute_fixed_width_partition<Hash, HasNulls, Partitioner, metadata_kind::packed32>(
      input,
      keys,
      num_partitions,
      seed,
      partitioner,
      *packed_config,
      fixed_width_indices,
      non_fixed_width_indices,
      stream,
      mr);
  }

  auto const unpacked_config =
    compute_launch_config<Hash, HasNulls, Partitioner, metadata_kind::unpacked32>(
      num_rows, num_partitions, max_fixed_width_size, has_fixed_width_payload);
  if (!unpacked_config) { return std::nullopt; }

  return execute_fixed_width_partition<Hash, HasNulls, Partitioner, metadata_kind::unpacked32>(
    input,
    keys,
    num_partitions,
    seed,
    partitioner,
    *unpacked_config,
    fixed_width_indices,
    non_fixed_width_indices,
    stream,
    mr);
}

template <template <typename> class Hash, bool HasNulls>
std::optional<fixed_width_partition_result> try_hash(table_view const& input,
                                                     table_view const& keys,
                                                     size_type num_partitions,
                                                     uint32_t seed,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  if (num_partitions > 0 && (num_partitions & (num_partitions - 1)) == 0) {
    return try_partitioner<Hash, HasNulls>(input,
                                           keys,
                                           num_partitions,
                                           seed,
                                           bitwise_partitioner<hash_value_type>{num_partitions},
                                           stream,
                                           mr);
  }
  return try_partitioner<Hash, HasNulls>(input,
                                         keys,
                                         num_partitions,
                                         seed,
                                         modulo_partitioner<hash_value_type>{num_partitions},
                                         stream,
                                         mr);
}

}  // namespace

std::optional<fixed_width_partition_result> try_fixed_width_hash_partition(
  table_view const& input,
  table_view const& keys,
  size_type num_partitions,
  hash_id hash_function,
  uint32_t seed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const has_nulls = cudf::has_nested_nulls(keys);
  switch (hash_function) {
    case hash_id::HASH_MURMUR3:
      return has_nulls ? try_hash<cudf::hashing::detail::MurmurHash3_x86_32, true>(
                           input, keys, num_partitions, seed, stream, mr)
                       : try_hash<cudf::hashing::detail::MurmurHash3_x86_32, false>(
                           input, keys, num_partitions, seed, stream, mr);
    case hash_id::HASH_IDENTITY:
      return has_nulls
               ? try_hash<identity_hash, true>(input, keys, num_partitions, seed, stream, mr)
               : try_hash<identity_hash, false>(input, keys, num_partitions, seed, stream, mr);
    default: return std::nullopt;
  }
}

}  // namespace cudf::detail
