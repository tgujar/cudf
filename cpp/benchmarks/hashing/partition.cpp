/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

static void bench_hash_partition(nvbench::state& state)
{
  using T = double;

  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols       = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));

  // Create owning columns
  auto input_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<T>()}, num_cols),
                                           row_count{static_cast<cudf::size_type>(num_rows)});
  auto input       = cudf::table_view(*input_table);

  auto columns_to_hash = std::vector<cudf::size_type>(num_cols);
  std::iota(columns_to_hash.begin(), columns_to_hash.end(), 0);

  // Set up CUDA stream for nvbench
  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto output = cudf::hash_partition(input, columns_to_hash, num_partitions);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  // Set memory usage statistics for nvbench
  state.add_global_memory_reads<T>(static_cast<int64_t>(num_rows) * num_cols);
  state.add_global_memory_writes<T>(static_cast<int64_t>(num_rows) * num_cols);
  state.add_global_memory_writes<cudf::size_type>(num_partitions);
}

NVBENCH_BENCH(bench_hash_partition)
  .set_name("hash_partition")
  .add_int64_axis("num_rows", {1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21})
  .add_int64_axis("num_cols", {1, 16, 256})
  .add_int64_axis("num_partitions", {64, 128, 256, 512, 1024});

namespace {

constexpr cudf::size_type fixed_num_rows = 1 << 21;
constexpr cudf::size_type fixed_num_cols = 8;

void run_partition(nvbench::state& state,
                   std::unique_ptr<cudf::table> const& input,
                   std::vector<cudf::size_type> const& keys,
                   cudf::size_type num_partitions)
{
  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto output = cudf::hash_partition(input->view(), keys, num_partitions);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_global_memory_reads<nvbench::int8_t>(static_cast<int64_t>(input->alloc_size()));
  state.add_global_memory_writes<nvbench::int8_t>(static_cast<int64_t>(input->alloc_size()));
  state.add_global_memory_writes<cudf::size_type>(num_partitions);
}

cudf::type_id fixed_width_type(std::string const& name)
{
  if (name == "int8") { return cudf::type_id::INT8; }
  if (name == "int16") { return cudf::type_id::INT16; }
  if (name == "int32") { return cudf::type_id::INT32; }
  if (name == "int64") { return cudf::type_id::INT64; }
  if (name == "float32") { return cudf::type_id::FLOAT32; }
  if (name == "float64") { return cudf::type_id::FLOAT64; }
  if (name == "timestamp_ns") { return cudf::type_id::TIMESTAMP_NANOSECONDS; }
  if (name == "decimal32") { return cudf::type_id::DECIMAL32; }
  if (name == "decimal64") { return cudf::type_id::DECIMAL64; }
  if (name == "decimal128") { return cudf::type_id::DECIMAL128; }
  throw std::invalid_argument{"Unknown fixed-width benchmark type: " + name};
}

void bench_partition_count_latency(nvbench::state& state)
{
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  auto input = create_sequence_table(cycle_dtypes({cudf::type_id::INT64}, fixed_num_cols),
                                     row_count{fixed_num_rows});
  run_partition(state, input, {0}, num_partitions);
}

void bench_fixed_width_families(nvbench::state& state)
{
  auto const type             = fixed_width_type(state.get_string("type"));
  auto const num_partitions   = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  auto const hash_all_columns = state.get_int64("hash_all_columns") != 0;
  auto const null_probability = state.get_float64("null_probability");

  data_profile const profile = data_profile_builder().null_probability(
    null_probability == 0.0 ? std::nullopt : std::optional<double>{null_probability});
  auto input =
    create_random_table(cycle_dtypes({type}, fixed_num_cols), row_count{fixed_num_rows}, profile);

  auto keys = std::vector<cudf::size_type>(hash_all_columns ? fixed_num_cols : 1);
  std::iota(keys.begin(), keys.end(), 0);
  run_partition(state, input, keys, num_partitions);
}

void bench_hybrid_payload(nvbench::state& state)
{
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  auto const types          = std::vector<cudf::type_id>{cudf::type_id::INT32,
                                                         cudf::type_id::INT8,
                                                         cudf::type_id::INT16,
                                                         cudf::type_id::INT64,
                                                         cudf::type_id::FLOAT64,
                                                         cudf::type_id::TIMESTAMP_NANOSECONDS,
                                                         cudf::type_id::DECIMAL128,
                                                         cudf::type_id::STRING};
  auto input =
    create_random_table(types, row_count{fixed_num_rows}, data_profile_builder().no_validity());
  run_partition(state, input, {0}, num_partitions);
}

void bench_equal_payload_column_count(nvbench::state& state)
{
  // Keep the fixed-width payload at 4 GiB while varying the column count. This isolates the copy
  // kernel's column batching from total payload size and hashes only one column.
  constexpr std::int64_t num_values = std::int64_t{1} << 29;
  auto const num_cols               = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_rows               = static_cast<cudf::size_type>(num_values / num_cols);
  auto input =
    create_sequence_table(cycle_dtypes({cudf::type_id::INT64}, num_cols), row_count{num_rows});
  run_partition(state, input, {0}, 1024);
}

NVBENCH_BENCH(bench_partition_count_latency)
  .set_name("hash_partition_partition_count_latency")
  .add_int64_axis("num_partitions",
                  {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384});

NVBENCH_BENCH(bench_fixed_width_families)
  .set_name("hash_partition_fixed_width_families")
  .add_string_axis("type",
                   {"int8",
                    "int16",
                    "int32",
                    "int64",
                    "float32",
                    "float64",
                    "timestamp_ns",
                    "decimal32",
                    "decimal64",
                    "decimal128"})
  .add_int64_axis("num_partitions", {64, 1024})
  .add_int64_axis("hash_all_columns", {0, 1})
  .add_float64_axis("null_probability", {0.0, 0.1});

NVBENCH_BENCH(bench_hybrid_payload)
  .set_name("hash_partition_hybrid_payload")
  .add_int64_axis("num_partitions", {8, 64, 1024, 4096});

NVBENCH_BENCH(bench_equal_payload_column_count)
  .set_name("hash_partition_equal_payload_column_count")
  .add_int64_axis("num_cols", {8, 256});

}  // namespace
