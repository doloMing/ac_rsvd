#pragma once

namespace ac_rsvd::math {

// log Phi_d(s), with Phi_d(s) = E exp(s d u_1^2) on the unit sphere.
double log_phi(int dimension, double s);

}  // namespace ac_rsvd::math
