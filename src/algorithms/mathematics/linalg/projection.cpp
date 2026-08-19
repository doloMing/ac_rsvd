#include "algorithms/mathematics/linalg/projection.hpp"

#include <stdexcept>

#include "algorithms/mathematics/linalg/blas_lapack.hpp"

namespace ac_rsvd {
namespace math {

void subtract_projection(const Matrix& basis, Matrix& block) {
    if (basis.rows() != block.rows()) {
        throw std::invalid_argument("Basis and block row counts do not match");
    }
    if (basis.cols() == 0 || block.cols() == 0) {
        return;
    }

    Matrix coefficients = transpose_multiply(basis, block);
    gemm(basis, false, coefficients, false, -1.0, 1.0, block);
}

void project_out(const Matrix& basis, Matrix& block, int passes) {
    for (int pass = 0; pass < passes; ++pass) {
        subtract_projection(basis, block);
    }
}

}  // namespace math
}  // namespace ac_rsvd
