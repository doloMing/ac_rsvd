#pragma once

#include <vector>

#include "algorithms/mathematics/linalg/matrix.hpp"

namespace ac_rsvd {
namespace math {

struct QrResult {
    Matrix q;
    Matrix r;
    std::vector<double> diagonal;
};

// This QR keeps the input column order and makes each nonzero diagonal positive.
QrResult thin_qr(const Matrix& matrix);

}  // namespace math
}  // namespace ac_rsvd
