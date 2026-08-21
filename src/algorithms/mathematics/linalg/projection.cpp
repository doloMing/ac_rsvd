#include "algorithms/mathematics/linalg/projection.hpp"

#include <stdexcept>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"

namespace ac_rsvd {
namespace math {
namespace {

void check_projection_dimensions(const Matrix& basis, const Matrix& block) {
    if (basis.rows() != block.rows()) {
        throw std::invalid_argument("Basis and block row counts do not match");
    }
}

void subtract_projection_with_workspace(
    const Matrix& basis,
    Matrix& block,
    Matrix& coefficients) {
    gemm(basis, true, block, false, 1.0, 0.0, coefficients);
    gemm(basis, false, coefficients, false, -1.0, 1.0, block);
}

}  // namespace

void subtract_projection(const Matrix& basis, Matrix& block) {
    check_projection_dimensions(basis, block);
    if (basis.cols() == 0 || block.cols() == 0) {
        return;
    }

    Matrix coefficients(basis.cols(), block.cols());
    subtract_projection_with_workspace(basis, block, coefficients);
}

void project_out(const Matrix& basis, Matrix& block, int passes) {
    check_projection_dimensions(basis, block);
    if (basis.cols() == 0 || block.cols() == 0 || passes <= 0) {
        return;
    }

    Matrix coefficients(basis.cols(), block.cols());
    // A second pass repairs most of the orthogonality lost to rounding.
    for (int pass = 0; pass < passes; ++pass) {
        subtract_projection_with_workspace(basis, block, coefficients);
    }
}

}  // namespace math
}  // namespace ac_rsvd
