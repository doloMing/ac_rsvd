#pragma once

#include "algorithms/mathematics/linalg/matrix.hpp"

namespace ac_rsvd {
namespace math {

void subtract_projection(const Matrix& basis, Matrix& block);
void project_out(const Matrix& basis, Matrix& block, int passes = 2);

}  // namespace math
}  // namespace ac_rsvd
