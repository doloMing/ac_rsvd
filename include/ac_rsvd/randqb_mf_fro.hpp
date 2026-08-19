#pragma once

#include <cstdint>

#include "ac_rsvd/factorization_result.hpp"
#include "ac_rsvd/matrix_operator.hpp"

namespace ac_rsvd {

struct RandQbMfFroOptions {
    // The paper's absolute Frobenius tolerance.
    double tolerance = 1e-8;
    int block_size = 32;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
};

FactorizationResult compute_randqb_mf_fro(
    const MatrixOperator& matrix,
    const RandQbMfFroOptions& options);

}  // namespace ac_rsvd
