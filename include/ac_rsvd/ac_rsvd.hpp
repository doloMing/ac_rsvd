#pragma once

#include <cstdint>

#include "ac_rsvd/factorization_result.hpp"
#include "ac_rsvd/matrix_operator.hpp"

namespace ac_rsvd {

struct AcRsvdOptions {
    double tolerance = 1e-8;
    double failure_probability = 1e-6;
    int block_size = 32;
    std::uint64_t seed = 0;
    std::uint64_t stream = 0;
};

FactorizationResult compute_ac_rsvd(
    const MatrixOperator& matrix,
    const AcRsvdOptions& options);

}  // namespace ac_rsvd
