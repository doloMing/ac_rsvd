#pragma once

#include "algorithms/mathematics/linalg/matrix.hpp"

namespace ac_rsvd {
namespace math {

void gemm(
    const Matrix& left,
    bool transpose_left,
    const Matrix& right,
    bool transpose_right,
    double alpha,
    double beta,
    Matrix& result);

// The three common products are AB, A^T B, and A B^T.
Matrix multiply(const Matrix& left, const Matrix& right);
Matrix transpose_multiply(const Matrix& left, const Matrix& right);
Matrix multiply_transpose(const Matrix& left, const Matrix& right);

double vector_norm(const double* values, int size);

}  // namespace math
}  // namespace ac_rsvd
