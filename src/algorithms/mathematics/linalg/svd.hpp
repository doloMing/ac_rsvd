#pragma once

#include <vector>

#include "algorithms/mathematics/linalg/matrix.hpp"

namespace ac_rsvd {
namespace math {

struct SvdResult {
    Matrix u;
    std::vector<double> singular_values;
    Matrix vt;
};

SvdResult thin_svd(const Matrix& matrix);

// B is the small matrix in A = Q B.
SvdResult compact_qb_svd(const Matrix& q, const Matrix& b);
SvdResult truncate_svd(const SvdResult& svd, int rank);

Matrix column_space(const Matrix& matrix, double relative_tolerance = -1.0);

}  // namespace math
}  // namespace ac_rsvd
