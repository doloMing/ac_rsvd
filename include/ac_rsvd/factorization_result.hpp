#pragma once

#include <vector>

namespace ac_rsvd {

enum class StopReason {
    certificate,
    tolerance_met,
    full_output_space,
    full_rank,
    final_input_direction,
    zero_residual
};

struct RunStatistics {
    long long directions_processed = 0;
    long long a_columns = 0;
    long long at_columns = 0;
    long long a_block_calls = 0;
    long long at_block_calls = 0;
};

struct FactorizationResult {
    int rows = 0;
    int cols = 0;
    int rank = 0;

    // U and V are column-major.
    std::vector<double> u;
    std::vector<double> singular_values;
    std::vector<double> v;

    RunStatistics statistics;
    StopReason stop_reason = StopReason::certificate;
    double residual_estimate_squared = 0.0;
    double residual_bound_squared = 0.0;
    double truncation_budget_squared = 0.0;
};

}  // namespace ac_rsvd
