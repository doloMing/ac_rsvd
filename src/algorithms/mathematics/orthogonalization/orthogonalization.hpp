#pragma once

#include <vector>

#include "algorithms/mathematics/linalg/matrix.hpp"
#include "algorithms/mathematics/linalg/qr.hpp"

namespace ac_rsvd {
namespace math {

struct BlockOrthogonalizationResult {
    // projected is the block after removing the old basis and before QR.
    Matrix projected;
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

ColumnOrthogonalizationResult orthogonalize_column(
    const Matrix& column,
    const Matrix& basis,
    double absolute_tolerance,
    int projection_passes = 2);

Matrix trailing_basis(
    const QrResult& block_qr,
    int first_column,
    double relative_tolerance = -1.0);

}  // namespace math
}  // namespace ac_rsvd
