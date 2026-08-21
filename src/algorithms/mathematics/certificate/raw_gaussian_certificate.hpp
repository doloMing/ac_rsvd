#pragma once

namespace ac_rsvd::math {

double chi_square_32_cdf(double value);
double chi_square_32_lower_quantile(double probability);

double raw_gaussian_log_psi(
    int dimension,
    double spectral_cap,
    double candidate,
    double gamma);

double choose_raw_gaussian_gamma(
    int dimension,
    double pilot_mean,
    double spectral_cap,
    double candidate);

double raw_gaussian_log_value(
    int count,
    double sum,
    int dimension,
    double spectral_cap,
    double gamma,
    double candidate);

bool raw_gaussian_crosses(
    int count,
    double sum,
    int dimension,
    double spectral_cap,
    double gamma,
    double candidate,
    double failure_probability);

double raw_gaussian_continuous_inverse(
    int count,
    double sum,
    int dimension,
    double spectral_cap,
    double gamma,
    double failure_probability,
    double upper);

}  // namespace ac_rsvd::math
