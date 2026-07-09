/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

#include <src/partitioning/fixed_width.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

using cudf::test::dictionary_column_wrapper;
using cudf::test::fixed_width_column_wrapper;
using cudf::test::lists_column_wrapper;
using cudf::test::strings_column_wrapper;
using cudf::test::structs_column_wrapper;

class FixedWidthHashPartitionTest : public cudf::test::BaseFixture {};

template <typename T>
class FixedWidthHashPartitionTypeTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedWidthHashPartitionTypeTest, cudf::test::FixedWidthTypes);

TYPED_TEST(FixedWidthHashPartitionTypeTest, CompatibilityAcceptsEveryFixedWidthType)
{
  auto column = cudf::make_empty_column(cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_TRUE(
    cudf::detail::is_fixed_width_partition_compatible(cudf::table_view{{column->view()}}));
}

TEST_F(FixedWidthHashPartitionTest, CompatibilityRejectsComplexAndDictionaryKeys)
{
  strings_column_wrapper strings{"a", "b"};
  lists_column_wrapper<int32_t> lists{{1, 2}, {3}};
  fixed_width_column_wrapper<int32_t> child{1, 2};
  structs_column_wrapper structs{{child}};
  dictionary_column_wrapper<int32_t> dictionary{1, 2};

  for (auto const& column : std::vector<cudf::column_view>{strings, lists, structs, dictionary}) {
    EXPECT_FALSE(cudf::detail::is_fixed_width_partition_compatible(cudf::table_view{{column}}));
  }
}

void expect_murmur_partitioned(cudf::table_view const& input,
                               std::vector<cudf::size_type> const& keys,
                               cudf::size_type num_partitions,
                               uint32_t seed = cudf::DEFAULT_HASH_SEED)
{
  auto [output, offsets] =
    cudf::hash_partition(input, keys, num_partitions, cudf::hash_id::HASH_MURMUR3, seed);

  ASSERT_EQ(offsets.size(), static_cast<std::size_t>(num_partitions + 1));
  EXPECT_EQ(offsets.front(), 0);
  EXPECT_EQ(offsets.back(), input.num_rows());
  EXPECT_TRUE(std::is_sorted(offsets.begin(), offsets.end()));
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(input, output->view());

  auto const sorted_input  = cudf::sort(input);
  auto const sorted_output = cudf::sort(output->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted_input->view(), sorted_output->view());

  auto hashes            = cudf::hashing::murmurhash3_x86_32(output->view().select(keys), seed);
  auto const host_hashes = cudf::test::to_host<uint32_t>(hashes->view()).first;
  for (cudf::size_type partition = 0; partition < num_partitions; ++partition) {
    for (auto row = offsets[partition]; row < offsets[partition + 1]; ++row) {
      EXPECT_EQ(host_hashes[row] % static_cast<uint32_t>(num_partitions),
                static_cast<uint32_t>(partition));
    }
  }
}

TEST_F(FixedWidthHashPartitionTest, PackedMetadataBoundaries)
{
  // 20 partition bits plus 12 bits for a 4 * 1024-row CTA exactly fills packed32.
  EXPECT_TRUE(cudf::detail::packed_metadata_fits(1 << 20, 4));
  // Five rows per thread need 13 offset bits, so the optimized unpacked layout is selected.
  EXPECT_FALSE(cudf::detail::packed_metadata_fits(1 << 20, 5));
  EXPECT_TRUE(cudf::detail::packed_metadata_fits(1, 1));
}

TEST_F(FixedWidthHashPartitionTest, FixedPointCompositeKeys)
{
  cudf::test::fixed_point_column_wrapper<int32_t> decimal32({11, -22, 33, 44, -55, 66, 77},
                                                            numeric::scale_type{-2});
  cudf::test::fixed_point_column_wrapper<int64_t> decimal64({101, 202, -303, 404, 505, -606, 707},
                                                            numeric::scale_type{3});
  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128(
    {1001, -2002, 3003, 4004, -5005, 6006, 7007}, numeric::scale_type{-7});
  fixed_width_column_wrapper<cudf::timestamp_ns, int64_t> timestamps{7, 6, 5, 4, 3, 2, 1};

  auto const input = cudf::table_view{{decimal32, decimal64, decimal128, timestamps}};
  expect_murmur_partitioned(input, {0, 1, 2, 3}, 7, 12345);
}

TEST_F(FixedWidthHashPartitionTest, MixedWidthsNullableAndLargePartitionCounts)
{
  constexpr cudf::size_type num_rows = 2053;
  auto const values                  = cuda::counting_iterator<int32_t>{0};
  auto const valid = thrust::make_transform_iterator(values, [](auto row) { return row % 7 != 0; });

  fixed_width_column_wrapper<int8_t, int32_t> bytes(values, values + num_rows);
  fixed_width_column_wrapper<int16_t, int32_t> shorts(values, values + num_rows);
  fixed_width_column_wrapper<int32_t> ints(values, values + num_rows, valid);
  fixed_width_column_wrapper<int64_t, int32_t> longs(values, values + num_rows);
  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128(
    values, values + num_rows, numeric::scale_type{-4});

  auto const input = cudf::table_view{{bytes, shorts, ints, longs, decimal128}};
  for (auto const partitions : {17, 1024, 1025, 4096}) {
    expect_murmur_partitioned(input, {0, 2, 4}, partitions);
  }
}

TEST_F(FixedWidthHashPartitionTest, CopiesWideFixedWidthTable)
{
  constexpr cudf::size_type num_rows = 257;
  constexpr std::size_t num_columns  = 33;

  std::vector<fixed_width_column_wrapper<int32_t>> columns;
  columns.reserve(num_columns);
  for (std::size_t column = 0; column < num_columns; ++column) {
    std::vector<int32_t> values(num_rows);
    std::iota(values.begin(), values.end(), static_cast<int32_t>(column * num_rows));
    columns.emplace_back(values.begin(), values.end());
  }

  std::vector<cudf::column_view> views;
  views.reserve(columns.size());
  std::transform(columns.begin(), columns.end(), std::back_inserter(views), [](auto const& column) {
    return static_cast<cudf::column_view>(column);
  });

  expect_murmur_partitioned(cudf::table_view{views}, {0}, 31);
}

TEST_F(FixedWidthHashPartitionTest, FloatingPointNormalizationAndSliceTail)
{
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  fixed_width_column_wrapper<double> doubles{
    0.0, -0.0, nan, -nan, 1.5, -2.25, 0.0, -0.0, nan, 9.0, 10.0};
  fixed_width_column_wrapper<int32_t> payload{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto const owner  = cudf::table_view{{doubles, payload}};
  auto const sliced = cudf::slice(owner, {1, 10}).front();

  expect_murmur_partitioned(sliced, {0}, 5, 9876);
}

TEST_F(FixedWidthHashPartitionTest, HybridPayloadColumnsStayAligned)
{
  fixed_width_column_wrapper<int32_t> row_id{0, 1, 2, 3, 4, 5, 6, 7};
  fixed_width_column_wrapper<int64_t> keys{17, 3, 91, 42, 8, 77, 11, 4};
  strings_column_wrapper strings({"zero", "one", "two", "three", "four", "five", "six", "seven"},
                                 {true, false, true, true, false, true, true, true});
  lists_column_wrapper<int32_t> lists{{0}, {1, 1}, {2}, {3, 3}, {4}, {5, 5}, {6}, {7, 7}};
  dictionary_column_wrapper<int32_t> dictionary{10, 11, 12, 13, 14, 15, 16, 17};
  cudf::test::fixed_point_column_wrapper<__int128_t> decimal128(
    {100, 101, 102, 103, 104, 105, 106, 107}, numeric::scale_type{-2});
  auto const input = cudf::table_view{{row_id, keys, strings, lists, dictionary, decimal128}};

  auto [output, offsets] = cudf::hash_partition(input, {1}, 5);
  auto const sorted      = cudf::sort_by_key(output->view(), output->view().select({0}));

  CUDF_TEST_EXPECT_TABLES_EQUAL(input, sorted->view());
  EXPECT_EQ(offsets.front(), 0);
  EXPECT_EQ(offsets.back(), input.num_rows());
}

TEST_F(FixedWidthHashPartitionTest, ExternalIdentityKeysWithNoFixedWidthPayload)
{
  constexpr cudf::size_type num_rows = 37;
  std::vector<std::string> values(num_rows);
  std::generate(values.begin(), values.end(), [row = 0]() mutable {
    return std::string{"row_"} + std::to_string(row++);
  });
  strings_column_wrapper strings(values.begin(), values.end());
  fixed_width_column_wrapper<int32_t> keys{0,  7,  1, 8, 2, 9, 3, 10, 4,  11, 5,  12, 6,
                                           13, 0,  7, 1, 8, 2, 9, 3,  10, 4,  11, 5,  12,
                                           6,  13, 0, 7, 1, 8, 2, 9,  3,  10, 4};
  auto const input = cudf::table_view{{strings}};

  constexpr cudf::size_type num_partitions = 7;
  auto [output, offsets]                   = cudf::hash_partition(
    input, cudf::table_view{{keys}}, num_partitions, cudf::hash_id::HASH_IDENTITY);
  auto const output_strings = cudf::test::to_host<std::string>(output->view().column(0)).first;
  auto const host_keys      = cudf::test::to_host<int32_t>(keys).first;

  for (cudf::size_type partition = 0; partition < num_partitions; ++partition) {
    for (auto row = offsets[partition]; row < offsets[partition + 1]; ++row) {
      auto const source_row = std::stoi(output_strings[row].substr(4));
      EXPECT_EQ(static_cast<uint32_t>(host_keys[source_row]) % num_partitions, partition);
    }
  }
}

}  // namespace
