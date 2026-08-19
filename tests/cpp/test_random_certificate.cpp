#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "algorithms/mathematics/certificate/certificate.hpp"
#include "algorithms/mathematics/certificate/spherical_moment.hpp"
#include "algorithms/mathematics/random/gaussian.hpp"
#include "algorithms/mathematics/random/philox.hpp"

namespace {

void check(bool condition, const std::string& message, int& failures) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

bool close(double first, double second, double tolerance) {
    return std::abs(first - second) <= tolerance;
}

void test_philox(int& failures) {
    std::array<std::uint32_t, 4> result =
        ac_rsvd::math::philox4x32({0, 0, 0, 0}, {0, 0});
    std::array<std::uint32_t, 4> expected = {
        0x6627e8d5U,
        0xe169c58dU,
        0xbc57ac4cU,
        0x9b00dbd8U};
    check(result == expected, "Philox reference vector", failures);
}

void test_gaussian(int& failures) {
    const int rows = 11;
    const int cols = 7;
    std::vector<double> values(rows * cols);
    ac_rsvd::math::fill_gaussian(values.data(), rows, cols, 19, 4, 23);

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            double expected = ac_rsvd::math::gaussian_value(
                19, 4, 23 + col, row);
            check(
                values[row + col * rows] == expected,
                "Gaussian indexing",
                failures);
        }
    }

    const int sample_count = 100000;
    std::vector<double> sample(sample_count);
    ac_rsvd::math::fill_gaussian(
        sample.data(), sample_count, 1, 71, 8, 0);

    double mean = 0.0;
    double second_moment = 0.0;
    for (double value : sample) {
        mean += value;
        second_moment += value * value;
    }
    mean /= sample_count;
    second_moment /= sample_count;

    check(std::abs(mean) < 0.02, "Gaussian mean", failures);
    check(
        std::abs(second_moment - 1.0) < 0.03,
        "Gaussian second moment",
        failures);
}

void test_log_phi(int& failures) {
    check(
        close(ac_rsvd::math::log_phi(1, -0.2), -0.2, 1e-15),
        "d=1 spherical moment",
        failures);
    check(
        close(ac_rsvd::math::log_phi(2, -1.0 / 3.0),
              -0.30574610888791818908, 2e-15),
        "d=2 spherical moment at -1/3",
        failures);
    check(
        close(ac_rsvd::math::log_phi(2, -1.0 / 12.0),
              -0.08159797516176963271, 2e-15),
        "d=2 spherical moment at -1/12",
        failures);
    check(
        close(ac_rsvd::math::log_phi(3, -1.0 / 3.0),
              -0.29192555287628617900, 2e-15),
        "d=3 spherical moment at -1/3",
        failures);
    check(
        close(ac_rsvd::math::log_phi(3, -1.0 / 12.0),
              -0.08060068275163107358, 2e-15),
        "d=3 spherical moment at -1/12",
        failures);
    check(
        close(ac_rsvd::math::log_phi(32, -1.0 / 3.0),
              -0.25916220595024251959, 2e-15),
        "d=32 spherical moment at -1/3",
        failures);
    check(
        close(ac_rsvd::math::log_phi(32, -1.0 / 12.0),
              -0.07753687815708648496, 2e-15),
        "d=32 spherical moment at -1/12",
        failures);
    check(
        close(ac_rsvd::math::log_phi(131072, -1.0 / 3.0),
              -0.25541372741052530308, 2e-15),
        "large-d spherical moment at -1/3",
        failures);
    check(
        close(ac_rsvd::math::log_phi(131072, -1.0 / 12.0),
              -0.07707545668906997158, 2e-15),
        "large-d spherical moment at -1/12",
        failures);
}

void test_certificate(int& failures) {
    ac_rsvd::math::Certificate certificate(32, 1.0, 0.1);

    for (int round = 0; round < 31 && !certificate.crossed(); ++round) {
        certificate.update(0.01, 32 - round);
    }

    check(certificate.crossed(), "certificate crossing", failures);
    check(certificate.trace().size() == 21, "certificate trace", failures);
    check(
        close(certificate.replay(1.0), certificate.value(), 1e-12),
        "online value matches replay",
        failures);

    double previous = certificate.replay(0.05);
    for (int index = 2; index <= 20; ++index) {
        double candidate = 0.05 * index;
        double current = certificate.replay(candidate);
        check(current >= previous, "certificate is monotone", failures);
        previous = current;
    }

    double inverse = certificate.continuous_inverse();
    check(inverse > 0.0 && inverse <= 1.0, "inverse range", failures);
    check(certificate.crosses(inverse), "inverse upper endpoint", failures);
    check(
        !certificate.crosses(inverse * (1.0 - 1e-10)),
        "point below inverse",
        failures);
}

}  // namespace

int main() {
    int failures = 0;
    test_philox(failures);
    test_gaussian(failures);
    test_log_phi(failures);
    test_certificate(failures);

    if (failures == 0) {
        std::cout << "random and certificate tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
