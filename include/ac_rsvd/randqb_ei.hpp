#pragma once

#include <cstdint>

#include "ac_rsvd/factorization_result.hpp"
#include "ac_rsvd/matrix_operator.hpp"

namespace ac_rsvd {

struct RandQbEiOptions {
    // This is the absolute-tolerance, zero-power version of randQB_EI.
    double tolerance = 1e-8;
    // The EI recurrence starts from this exact Frobenius norm.
    double frobenius_norm = -1.0;
    int block_size = 16;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
};

FactorizationResult compute_randqb_ei(
    const MatrixOperator& matrix,
    const RandQbEiOptions& options);

}  // namespace ac_rsvd
