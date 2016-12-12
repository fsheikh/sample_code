#include <limits.h>
#include <stdint.h>
#include "algorithm_test.h"
#include "gtest/gtest.h"

// Sample Application for GoogleTest
// Build and Run
// make gtest
// make gtest_sample

TEST(AlignedRegions, NumRegions) {

    std::pair<uint32_t, uint32_t> input_pair(17, 6);
    static constexpr unsigned EXPECTED_REGIONS {4};
    unsigned call_count = 0;
    auto sfunc = [&call_count](uint32_t x, uint32_t y) {
                   call_count++;
                   return;
                 };
    for_aligned_regions(input_pair.first, input_pair.second, sfunc);
    EXPECT_EQ(EXPECTED_REGIONS, call_count);
}

TEST(AlignedRegions, CorrectIntervals) {

    using PairType = std::pair<uint32_t, uint32_t>;
    PairType input_pair(17,6);
    PairType Expected_Intervals[] = {PairType(17,0), PairType(18,1), PairType(20,1), PairType(22, 0)};
    unsigned interval_bitmap = 0;
    unsigned interval_index  = 0;

    auto sfunc = [&interval_bitmap, &interval_index, Expected_Intervals] (uint32_t x, uint32_t y) {

                     if (x == Expected_Intervals[interval_index].first &&
                         y == Expected_Intervals[interval_index].second) {

                         interval_bitmap |= (1u << interval_index);
                     }

                     interval_index++;
                     return;
                };

    for_aligned_regions(input_pair.first, input_pair.second, sfunc);
    // Lower four bit should be set in interval_bitmap
    EXPECT_EQ(0xF, interval_bitmap);
}
