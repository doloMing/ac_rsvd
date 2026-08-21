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

enum class ResidualBoundSource {
    none,
    global_certificate,
    diagnostic_certificate,
    diagnostic_spectral_cap,
    exact_residual
};

enum class FactorizationStatus {
    // Factor arrays contain the requested approximation.
    success = 0,
    // Exact terminal validation found that the fixed range is insufficient.
    certificate_miss = 1
};

struct DiagnosticTrace {
    bool attempted = false;
    bool heldout_activated = false;
    bool crossed = false;
    bool reached_refinement_target = false;

    int trigger_range_rank = 0;
    int pilot_directions = 0;
    // All held-out columns revealed through the final generated block.
    int heldout_directions = 0;
    int heldout_endpoints = 0;
    int first_crossing_direction = 0;
    // Prefix that attains the returned running-minimum bound.
    int bound_direction = 0;

    double trigger_mean_ratio = 0.0;
    double pilot_mean = 0.0;
    double spectral_cap = 0.0;
    double gamma = 0.0;
    double first_crossing_sum = 0.0;
    double bound_sum = 0.0;
    double residual_bound_squared = 0.0;
};

struct RunStatistics {
    long long directions_processed = 0;
    long long a_columns = 0;
    long long at_columns = 0;
    long long a_block_calls = 0;
    long long at_block_calls = 0;
    long long assimilated_directions = 0;
    long long validation_directions = 0;
    // This trace prefix gives residual_bound_squared.
    long long certificate_bound_round = 0;

    long long ordinary_columns = 0;
    long long ordinary_block_calls = 0;
    long long ordinary_observations = 0;
    long long ordinary_assimilated = 0;
    long long ordinary_discarded = 0;

    long long diagnostic_columns = 0;
    long long diagnostic_block_calls = 0;
    long long diagnostic_pilot_columns = 0;
    long long diagnostic_heldout_columns = 0;

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
    // Restart weights are valid through this certificate round.
    int certificate_max_observations = 0;

    // U and V are column-major.
    std::vector<double> u;
    std::vector<double> singular_values;
    std::vector<double> v;

    RunStatistics statistics;
    DiagnosticTrace diagnostic;
    StopReason stop_reason = StopReason::certificate;
    // A miss has no U, singular values, or V result.
    FactorizationStatus status = FactorizationStatus::success;
    ResidualBoundSource residual_bound_source = ResidualBoundSource::none;
    // This distinguishes optional exact truncation from the default path.
    bool exact_frobenius_side_information_used = false;

    // Residual estimate; exact-Fro mode stores the terminal projection residual.
    // EI may make its estimator negative by cancellation.
    double residual_estimate_squared = 0.0;
    // AC-RSVD's exact-model continuous inverse U_t.
    double residual_bound_squared = 0.0;
    // Squared tail budget used for the final compact SVD.
    double truncation_budget_squared = 0.0;
    // The certified budget before a terminal range correction.
    double base_truncation_budget_squared = 0.0;
    // Squared decrease from the useful suffix of the terminal block.
    double residual_decrease_squared = 0.0;
    // Independent error checks set this after the timed algorithm has ended.
    double direct_error_squared = -1.0;

    // These arrays store the scalar certificate path.
    std::vector<double> certificate_observations;
    std::vector<int> certificate_dimensions;
    std::vector<double> certificate_leakages;
    std::vector<double> certificate_scales;
};

}  // namespace ac_rsvd
