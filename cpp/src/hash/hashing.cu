/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/detail/hashing.hpp>
#include <cudf/copying.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/detail/scatter.hpp>

#include <thrust/tabulate.h>

namespace cudf {

namespace {

constexpr size_type BLOCK_SIZE = 512;
constexpr size_type ROWS_PER_THREAD = 8;
constexpr size_type ELEMENT_PER_BLOCK = 2;

/** 
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses the modulo operation.
 */
template <typename hash_value_t>
class modulo_partitioner
{
 public:
  modulo_partitioner(size_type num_partitions) : divisor{num_partitions} {}

  __device__
  size_type operator()(hash_value_t hash_value) const {
    return hash_value % divisor;
  }

 private:
  const size_type divisor;
};

template <typename T>
bool is_power_two(T number) {
  return (0 == (number & (number - 1)));
}

/** 
 * @brief  Functor to map a hash value to a particular 'bin' or partition number
 * that uses a bitwise mask. Only works when num_partitions is a power of 2.
 *
 * For n % d, if d is a power of two, then it can be computed more efficiently via 
 * a single bitwise AND as:
 * n & (d - 1)
 */
template <typename hash_value_t>
class bitwise_partitioner
{
 public:
  bitwise_partitioner(size_type num_partitions) : mask{(num_partitions - 1)} {
    assert(is_power_two(num_partitions));
  }

  __device__
  size_type operator()(hash_value_t hash_value) const {
    return hash_value & mask; // hash_value & (num_partitions - 1)
  }

 private:
  const size_type mask;
};

/* --------------------------------------------------------------------------*/
/** 
 * @brief Computes which partition each row of a device_table will belong to based
   on hashing each row, and applying a partition function to the hash value. 
   Records the size of each partition for each thread block as well as the global
   size of each partition across all thread blocks.
 * 
 * @param[in] the_table The table whose rows will be partitioned
 * @param[in] num_rows The number of rows in the table
 * @param[in] num_partitions The number of partitions to divide the rows into
 * @param[in] the_partitioner The functor that maps a rows hash value to a partition number
 * @param[out] row_partition_numbers Array that holds which partition each row belongs to
 * @param[out] row_partition_offset Array that holds the offset of each row in its partition of
 * the thread block
 * @param[out] block_partition_sizes Array that holds the size of each partition for each block,
 * i.e., { {block0 partition0 size, block1 partition0 size, ...}, 
         {block0 partition1 size, block1 partition1 size, ...},
         ...
         {block0 partition(num_partitions-1) size, block1 partition(num_partitions -1) size, ...} }
 * @param[out] global_partition_sizes The number of rows in each partition.
 */
/* ----------------------------------------------------------------------------*/
template <class row_hasher_t, typename partitioner_type>
__global__
void compute_row_partition_numbers(row_hasher_t the_hasher,
                                   const size_type num_rows,
                                   const size_type num_partitions,
                                   const partitioner_type the_partitioner,
                                   size_type * row_partition_numbers,
                                   size_type * row_partition_offset,
                                   size_type * block_partition_sizes,
                                   size_type * global_partition_sizes)
{
  // Accumulate histogram of the size of each partition in shared memory
  extern __shared__ size_type shared_partition_sizes[];

  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Initialize local histogram
  size_type partition_number = threadIdx.x;
  while(partition_number < num_partitions)
  {
    shared_partition_sizes[partition_number] = 0;
    partition_number += blockDim.x;
  }

  __syncthreads();

  // Compute the hash value for each row, store it to the array of hash values
  // and compute the partition to which the hash value belongs and increment
  // the shared memory counter for that partition
  while( row_number < num_rows)
  {
    const hash_value_type row_hash_value = the_hasher(row_number);

    const size_type partition_number = the_partitioner(row_hash_value);

    row_partition_numbers[row_number] = partition_number;

    row_partition_offset[row_number] = atomicAdd(
      &(shared_partition_sizes[partition_number]), size_type(1));

    row_number += blockDim.x * gridDim.x;
  }

  __syncthreads();

  // Flush shared memory histogram to global memory
  partition_number = threadIdx.x;
  while(partition_number < num_partitions)
  {
    const size_type block_partition_size = shared_partition_sizes[partition_number];

    // Update global size of each partition
    atomicAdd(&global_partition_sizes[partition_number], block_partition_size);

    // Record the size of this partition in this block
    const size_type write_location = partition_number * gridDim.x + blockIdx.x;
    block_partition_sizes[write_location] = block_partition_size;
    partition_number += blockDim.x;
  }
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief Move one column from the input table to the hashed table.
 * 
 * @param[in] input_buf Data buffer of the column in the input table
 * @param[out] output_buf Preallocated data buffer of the column in the output table
 * @param[in] num_rows The number of rows in each column
 * @param[in] num_partitions The number of partitions to divide the rows into
 * @param[in] row_partition_numbers Array that holds which partition each row belongs to
 * @param[in] row_partition_offset Array that holds the offset of each row in its partition of
 * the thread block.
 * @param[in] block_partition_sizes Array that holds the size of each partition for each block
 * @param[in] scanned_block_partition_sizes The scan of block_partition_sizes
 */
/* ----------------------------------------------------------------------------*/
template <typename DataType>
__global__
void move_to_output_buffer(DataType const *input_buf,
                           DataType *output_buf,
                           const size_type num_rows,
                           const size_type num_partitions,
                           size_type * row_partition_numbers,
                           size_type * row_partition_offset,
                           size_type * block_partition_sizes,
                           size_type * scanned_block_partition_sizes)
{
  extern __shared__ char shared_memory[];
  DataType *block_output = (DataType *)shared_memory;
  size_type *partition_offset_shared = (size_type *)(block_output + BLOCK_SIZE * ROWS_PER_THREAD);
  size_type *partition_offset_global = (size_type *)(partition_offset_shared + num_partitions + 1);

  size_type ipartition;

  typedef cub::BlockScan<size_type, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // use ELEMENT_PER_BLOCK=2 to support upto 1024 partitions 
  size_type temp_histo[ELEMENT_PER_BLOCK];

  for (int i = 0; i < ELEMENT_PER_BLOCK; ++i) {
    if (ELEMENT_PER_BLOCK * threadIdx.x + i < num_partitions) {
      temp_histo[i] = block_partition_sizes[blockIdx.x + (ELEMENT_PER_BLOCK * threadIdx.x + i) * gridDim.x]; 
    } else {
      temp_histo[i] = 0;
    }
  }

  __syncthreads();

  BlockScan(temp_storage).InclusiveSum(temp_histo, temp_histo);

  __syncthreads();

  if (threadIdx.x == 0) {
    partition_offset_shared[0] = 0;
  }

  for (int i = 0; i < ELEMENT_PER_BLOCK; ++i) {
    if (ELEMENT_PER_BLOCK * threadIdx.x + i < num_partitions) {
      partition_offset_shared[ELEMENT_PER_BLOCK * threadIdx.x + i + 1] = temp_histo[i]; 
    } 
  }

  // Fetch the offset in the output buffer of each partition in this thread block
  ipartition = threadIdx.x;
  while (ipartition < num_partitions) {
    partition_offset_global[ipartition] = scanned_block_partition_sizes[ipartition * gridDim.x + blockIdx.x];
    ipartition += blockDim.x;
  }

  __syncthreads();

  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  // Fetch the input data to shared memory
  while ( row_number < num_rows ) {
    ipartition = row_partition_numbers[row_number];

    block_output[
      partition_offset_shared[ipartition] + row_partition_offset[row_number]
    ] = input_buf[row_number];

    row_number += blockDim.x * gridDim.x;
  }

  __syncthreads();

  // Copy data from shared memory to output buffer

  constexpr int nthreads_partition = 16; // Use 16 threads to copy each partition, assume BLOCK_SIZE
                                         // is divisible by 16
  
  for (ipartition = threadIdx.x / nthreads_partition;
       ipartition < num_partitions;
       ipartition += BLOCK_SIZE / nthreads_partition) {
    
    size_type row_offset = threadIdx.x % nthreads_partition;
    size_type nelements_partition = partition_offset_shared[ipartition + 1] - partition_offset_shared[ipartition];

    while (row_offset < nelements_partition) {
      output_buf[partition_offset_global[ipartition] + row_offset]
        = block_output[partition_offset_shared[ipartition] + row_offset];
      
      row_offset += nthreads_partition;
    }
  }
}

struct move_to_output_buffer_dispatcher{
  template <typename DataType,
      std::enable_if_t<is_fixed_width<DataType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                  const size_type num_partitions,
                  size_type * row_partition_numbers,
                  size_type * row_partition_offset,
                  size_type * block_partition_sizes,
                  size_type * scanned_block_partition_sizes,
                  size_type grid_size,
                  rmm::mr::device_memory_resource* mr,
                  cudaStream_t stream)
  {
    CUDF_EXPECTS(input.null_mask() == nullptr, "null input column unsupported");

    rmm::device_buffer output(input.size() * sizeof(DataType), stream, mr);

    int const smem = BLOCK_SIZE * ROWS_PER_THREAD * sizeof(DataType)
      + (num_partitions + 1) * sizeof(size_type) * 2;
    move_to_output_buffer<DataType>
      <<<grid_size, BLOCK_SIZE, smem, stream>>>(
        input.data<DataType>(), static_cast<DataType*>(output.data()), input.size(),
        num_partitions, row_partition_numbers, row_partition_offset,
        block_partition_sizes, scanned_block_partition_sizes
    );

    return std::make_unique<column>(input.type(), input.size(), std::move(output));
  }

  template <typename DataType,
      std::enable_if_t<not is_fixed_width<DataType>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                  const size_type num_partitions,
                  size_type * row_partition_numbers,
                  size_type * row_partition_offset,
                  size_type * block_partition_sizes,
                  size_type * scanned_block_partition_sizes,
                  size_type grid_size,
                  rmm::mr::device_memory_resource* mr,
                  cudaStream_t stream)
  {
    CUDF_FAIL("non-fixed width types unsupported");
  }
};

template <bool has_nulls>
std::pair<std::unique_ptr<experimental::table>, std::vector<size_type>>
hash_partition_table(table_view const& input,
                     table_view const &table_to_hash,
                     const size_type num_partitions,
                     rmm::mr::device_memory_resource* mr,
                     cudaStream_t stream)
{
  auto const num_rows = table_to_hash.num_rows();

  constexpr size_type rows_per_block = BLOCK_SIZE * ROWS_PER_THREAD;
  auto grid_size = util::div_rounding_up_safe(num_rows, rows_per_block);

  // Allocate array to hold which partition each row belongs to
  auto row_partition_numbers = rmm::device_vector<size_type>(num_rows);

  // Array to hold the size of each partition computed by each block
  //  i.e., { {block0 partition0 size, block1 partition0 size, ...}, 
  //          {block0 partition1 size, block1 partition1 size, ...},
  //          ...
  //          {block0 partition(num_partitions-1) size, block1 partition(num_partitions -1) size, ...} }
  auto block_partition_sizes = rmm::device_vector<size_type>(grid_size * num_partitions);

  auto scanned_block_partition_sizes = rmm::device_vector<size_type>(grid_size * num_partitions);

  // Holds the total number of rows in each partition
  auto global_partition_sizes = rmm::device_vector<size_type>(num_partitions, size_type{0});

  auto row_partition_offset = rmm::device_vector<size_type>(num_rows);

  auto const device_input = table_device_view::create(table_to_hash, stream);
  auto const hasher = experimental::row_hasher<MurmurHash3_32, has_nulls>(*device_input);

  // If the number of partitions is a power of two, we can compute the partition 
  // number of each row more efficiently with bitwise operations
  if (is_power_two(num_partitions)) {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = bitwise_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers
        <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(size_type), stream>>>(
            hasher, num_rows, num_partitions,
            partitioner_type(num_partitions),
            row_partition_numbers.data().get(),
            row_partition_offset.data().get(),
            block_partition_sizes.data().get(),
            global_partition_sizes.data().get());
  } else {
    // Determines how the mapping between hash value and partition number is computed
    using partitioner_type = modulo_partitioner<hash_value_type>;

    // Computes which partition each row belongs to by hashing the row and performing
    // a partitioning operator on the hash value. Also computes the number of
    // rows in each partition both for each thread block as well as across all blocks
    compute_row_partition_numbers
        <<<grid_size, BLOCK_SIZE, num_partitions * sizeof(size_type), stream>>>(
            hasher, num_rows, num_partitions,
            partitioner_type(num_partitions),
            row_partition_numbers.data().get(),
            row_partition_offset.data().get(),
            block_partition_sizes.data().get(),
            global_partition_sizes.data().get());
  }

  // Compute exclusive scan of all blocks' partition sizes in-place to determine 
  // the starting point for each blocks portion of each partition in the output
  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         block_partition_sizes.begin(), 
                         block_partition_sizes.end(), 
                         scanned_block_partition_sizes.data().get());

  // Compute exclusive scan of size of each partition to determine offset location
  // of each partition in final output.
  // TODO This can be done independently on a separate stream
  size_type * scanned_global_partition_sizes{global_partition_sizes.data().get()};
  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         global_partition_sizes.begin(), 
                         global_partition_sizes.end(),
                         scanned_global_partition_sizes);

  // Copy the result of the exlusive scan to the output offsets array
  // to indicate the starting point for each partition in the output
  std::vector<size_type> partition_offsets(num_partitions);
  CUDA_TRY(cudaMemcpyAsync(partition_offsets.data(), 
                           scanned_global_partition_sizes, 
                           num_partitions * sizeof(size_type),
                           cudaMemcpyDeviceToHost,
                           stream));

  std::vector<std::unique_ptr<column>> output_cols(input.num_columns());

  // Move data from input table to hashed table
  auto row_partition_numbers_ptr {row_partition_numbers.data().get()};
  auto row_partition_offset_ptr {row_partition_offset.data().get()};
  auto block_partition_sizes_ptr {block_partition_sizes.data().get()};
  auto scanned_block_partition_sizes_ptr {scanned_block_partition_sizes.data().get()};
  std::transform(input.begin(), input.end(), output_cols.begin(),
    [=](auto const& col) {
      return cudf::experimental::type_dispatcher(
        col.type(),
        move_to_output_buffer_dispatcher{},
        col,
        num_partitions,
        row_partition_numbers_ptr,
        row_partition_offset_ptr,
        block_partition_sizes_ptr,
        scanned_block_partition_sizes_ptr,
        grid_size, mr, stream);
    });

  auto output {std::make_unique<experimental::table>(std::move(output_cols))};
  return std::make_pair(std::move(output), std::move(partition_offsets));
}

struct nvtx_raii {
  nvtx_raii(char const* name, nvtx::color color) { nvtx::range_push(name, color); }
  ~nvtx_raii() { nvtx::range_pop(); }
};

}  // namespace

namespace detail {

std::pair<std::unique_ptr<experimental::table>, std::vector<size_type>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream)
{
  // Push/pop nvtx range around the scope of this function
  nvtx_raii("CUDF_HASH_PARTITION", nvtx::PARTITION_COLOR);

  auto table_to_hash = input.select(columns_to_hash);

  // Return empty result if there are no partitions or nothing to hash
  if (num_partitions <= 0 || input.num_rows() == 0 || table_to_hash.num_columns() == 0) {
    return std::make_pair(experimental::empty_like(input), std::vector<size_type>{});
  }

  if (has_nulls(table_to_hash)) {
    return hash_partition_table<true>(
        input, table_to_hash, num_partitions, mr, stream);
  } else {
    return hash_partition_table<false>(
        input, table_to_hash, num_partitions, mr, stream);
  }
}

std::unique_ptr<column> hash(table_view const& input,
                             std::vector<uint32_t> const& initial_hash,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream)
{
  // TODO this should be UINT32
  auto output = make_numeric_column(data_type(INT32), input.num_rows());

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    return output;
  }

  bool const nullable = has_nulls(input);
  auto const device_input = table_device_view::create(input, stream);
  auto output_view = output->mutable_view();

  // Compute the hash value for each row depending on the specified hash function
  if (!initial_hash.empty()) {
    CUDF_EXPECTS(initial_hash.size() == size_t(input.num_columns()),
      "Expected same size of initial hash values as number of columns");
    auto device_initial_hash = rmm::device_vector<uint32_t>(initial_hash);

    if (nullable) {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          output_view.begin<int32_t>(), output_view.end<int32_t>(),
          experimental::row_hasher_initial_values<MurmurHash3_32, true>(
              *device_input, device_initial_hash.data().get()));
    } else {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          output_view.begin<int32_t>(), output_view.end<int32_t>(),
          experimental::row_hasher_initial_values<MurmurHash3_32, false>(
              *device_input, device_initial_hash.data().get()));
    }
  } else {
    if (nullable) {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          output_view.begin<int32_t>(), output_view.end<int32_t>(),
          experimental::row_hasher<MurmurHash3_32, true>(*device_input));
    } else {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          output_view.begin<int32_t>(), output_view.end<int32_t>(),
          experimental::row_hasher<MurmurHash3_32, false>(*device_input));
    }
  }

  return output;
}

}  // namespace detail

std::pair<std::unique_ptr<experimental::table>, std::vector<size_type>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions,
               rmm::mr::device_memory_resource* mr)
{
  return detail::hash_partition(input, columns_to_hash, num_partitions, mr);
}

std::unique_ptr<column> hash(table_view const& input,
                             std::vector<uint32_t> const& initial_hash,
                             rmm::mr::device_memory_resource* mr)
{
  return detail::hash(input, initial_hash, mr);
}

}  // namespace cudf
