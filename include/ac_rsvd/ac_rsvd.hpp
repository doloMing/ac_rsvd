#pragma once

#include <cstdint>
#include <optional>

#include "ac_rsvd/factorization_result.hpp"
#include "ac_rsvd/matrix_operator.hpp"

namespace ac_rsvd {

// This entry point evaluates the paper's Section 3 exact-model path in FP64.
struct AcRsvdOptions {
    // Absolute Frobenius tolerance and total failure probability.
    double tolerance = 1e-8;
    double failure_probability = 1e-6;

    // Fixed empirical default; blocking does not change the ordered decisions.
    int block_size = 16;
    bool use_enhanced_mode = true;
    // Fixed empirical ceiling for held-out directions in the enhanced mode.
    int diagnostic_test_size = 512;
    // Seed and stream identify the ordered Gaussian direction sequence.
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;

    // Optional exact side information for terminal Frobenius truncation.
    std::optional<double> exact_frobenius_norm_squared;

    // This keeps block operator calls but uses column-wise QR for the ablation.
    bool use_sequential_orthogonalization = false;
};

FactorizationResult compute_ac_rsvd(
    const MatrixOperator& matrix,
    const AcRsvdOptions& options);

}  // namespace ac_rsvd
