#pragma once

#include <vector>

namespace ac_rsvd {
namespace math {

struct AnalyticTheoryBound {
    int directions = 0;
    int forward_columns = 0;
    int forward_block_calls = 0;
    int k = -1;
    int p = -1;
    int range_directions = -1;
    int appendix_e5_delay = -1;
    int tolerance_rank = -1;
    bool uses_deterministic_cap = true;
};

struct AnalyticTheoryBounds {
    int deterministic_cap = 0;
    AnalyticTheoryBound theorem2_e5;
    AnalyticTheoryBound corollary21_e5;
};

int deterministic_direction_cap(
    int output_dimension,
    int input_dimension,
    int matrix_rank);

int appendix_e5_certificate_delay(
    int input_dimension,
    int range_directions,
    double failure_probability,
    double rho,
    double certificate_failure_probability);

int block_rounded_columns(
    int input_dimension,
    int block_size,
    int directions);

int block_call_bound(int block_size, int directions);

// These bounds use the analytic delay from Appendix E.5.
AnalyticTheoryBounds evaluate_theory_bounds_e5(
    const std::vector<double>& singular_values,
    int output_dimension,
    int input_dimension,
    double tolerance,
    double failure_probability,
    double range_failure_probability,
    double certificate_failure_probability,
    double rho,
    double alpha,
    int block_size);

}  // namespace math
}  // namespace ac_rsvd
