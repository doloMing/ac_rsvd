#pragma once

#include <vector>

#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/linalg/qr.hpp"

namespace ac_rsvd {
namespace math {

struct BlockOrthogonalizationResult {
    Matrix q;
    Matrix r;
    std::vector<double> diagonal;
};

struct ColumnOrthogonalizationResult {
    Matrix q;
    double norm = 0.0;
    bool accepted = false;
};

BlockOrthogonalizationResult orthogonalize_block(
    const Matrix& block,
    const Matrix& basis,
    int projection_passes = 2);

void orthogonalize_block_in_place(
    Matrix& block,
    const Matrix& basis,
    Matrix& r,
    std::vector<double>& diagonal,
    int projection_passes = 2);

BlockOrthogonalizationResult orthogonalize_sequential_block(
    const Matrix& block,
    const Matrix& basis,
    int projection_passes = 2);

ColumnOrthogonalizationResult orthogonalize_column(
    const Matrix& column,
    const Matrix& basis,
    double absolute_tolerance,
    int projection_passes = 2);

Matrix trailing_basis(
    const Matrix& q,
    const Matrix& r,
    int first_column,
    double relative_tolerance = -1.0);

}  // namespace math
}  // namespace ac_rsvd
