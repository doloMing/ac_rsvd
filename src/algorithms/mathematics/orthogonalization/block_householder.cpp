#include "algorithms/mathematics/orthogonalization/orthogonalization.hpp"

#include "algorithms/mathematics/linalg/projection.hpp"

namespace ac_rsvd {
namespace math {

BlockOrthogonalizationResult orthogonalize_block(
    const Matrix& block,
    const Matrix& basis,
    int projection_passes) {
    BlockOrthogonalizationResult result;
    result.projected = block;
    project_out(basis, result.projected, projection_passes);

    QrResult qr = thin_qr(result.projected);
    result.q = qr.q;
    result.r = qr.r;
    result.diagonal = qr.diagonal;
    return result;
}

}  // namespace math
}  // namespace ac_rsvd
