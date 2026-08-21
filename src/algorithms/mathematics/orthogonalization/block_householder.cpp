#include "algorithms/mathematics/orthogonalization/orthogonalization.hpp"

#include "algorithms/mathematics/linalg/projection.hpp"

namespace ac_rsvd {
namespace math {

BlockOrthogonalizationResult orthogonalize_block(
    const Matrix& block,
    const Matrix& basis,
    int projection_passes) {
    BlockOrthogonalizationResult result;
    result.q = block;
    orthogonalize_block_in_place(
        result.q,
        basis,
        result.r,
        result.diagonal,
        projection_passes);
    return result;
}

void orthogonalize_block_in_place(
    Matrix& block,
    const Matrix& basis,
    Matrix& r,
    std::vector<double>& diagonal,
    int projection_passes) {
    // Reorthogonalize first, then keep the ordered Householder QR factors.
    project_out(basis, block, projection_passes);
    thin_qr_in_place(block, r, diagonal);
}

}  // namespace math
}  // namespace ac_rsvd
