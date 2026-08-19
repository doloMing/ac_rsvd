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

    double total_seconds = 0.0;
    double a_seconds = 0.0;
    double at_seconds = 0.0;
    double orthogonalization_seconds = 0.0;
    double certificate_seconds = 0.0;
    double svd_seconds = 0.0;
};

struct FactorizationResult {
    int rows = 0;
    int cols = 0;
    int rank = 0;
    // Dimension of the final range before compact SVD truncation.
    int range_rank = 0;
    // AC-RSVD rank from the stopping boundary without continuous inversion.
    int boundary_only_rank = 0;

    // U and V are column-major.
    std::vector<double> u;
    std::vector<double> singular_values;
    std::vector<double> v;

    RunStatistics statistics;
    StopReason stop_reason = StopReason::certificate;

    // Baseline error diagnostic. EI may make it negative by cancellation.
    double residual_estimate_squared = 0.0;
    // AC-RSVD's exact-model continuous inverse U_t.
    double residual_bound_squared = 0.0;
    // Squared tail budget used for the final compact SVD.
    double truncation_budget_squared = 0.0;
    // Squared decrease from the useful suffix of the terminal block.
    double residual_decrease_squared = 0.0;
    // Independent error checks set this after the timed algorithm has ended.
    double direct_error_squared = -1.0;

    // These two arrays store the certificate pairs (X_t, d_t).
    std::vector<double> certificate_observations;
    std::vector<int> certificate_dimensions;
};

}  // namespace ac_rsvd
