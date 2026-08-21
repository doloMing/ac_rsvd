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

struct TallSvdResult {
    std::vector<double> singular_values;
    Matrix vt;
    int selected_rank = 0;
    bool used_gram = false;
    bool exact_frobenius_budget_used = false;
    bool exact_frobenius_budget_feasible = true;
    double projected_norm_squared = 0.0;
    double residual_norm_squared = 0.0;
    double tail_budget_squared = 0.0;
};

SvdResult thin_svd(const Matrix& matrix);

// Overwrite a symmetric matrix and return its largest eigenvalue.
double largest_symmetric_eigenvalue_in_place(Matrix& matrix);

// This stable path is the fallback for strict or ill-conditioned work.
TallSvdResult stable_tall_svd_in_place(Matrix& matrix);

// Overwrite a tall matrix with all left singular vectors.
TallSvdResult tall_svd_in_place(Matrix& matrix);

// Overwrite a tall matrix with the left vectors selected by the tail budget.
TallSvdResult tall_svd_in_place(Matrix& matrix, double tail_budget);

// Derive the tail budget from an exact matrix Frobenius norm.
TallSvdResult tall_svd_exact_frobenius_in_place(
    Matrix& matrix,
    double matrix_norm_squared,
    double tolerance_squared);

// B is the small matrix in A = Q B.
SvdResult compact_qb_svd(const Matrix& q, const Matrix& b);
SvdResult truncate_svd(const SvdResult& svd, int rank);

// Return the left singular vectors whose singular values clear the cutoff.
Matrix column_space(
    const Matrix& matrix,
    double relative_tolerance = -1.0,
    double absolute_tolerance = 0.0);

}  // namespace math
}  // namespace ac_rsvd
