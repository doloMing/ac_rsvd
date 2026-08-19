#include "algorithms/mathematics/random/gaussian.hpp"

#include <array>
#include <cmath>
#include <cstdint>

#include "algorithms/mathematics/random/philox.hpp"

namespace ac_rsvd::math {
namespace {

constexpr double two_pi = 6.283185307179586476925286766559;
constexpr double uint32_scale = 1.0 / 4294967296.0;

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::array<std::uint32_t, 2> make_key(
    std::uint64_t seed,
    std::uint64_t stream) {
    std::uint64_t mixed = splitmix64(seed);
    mixed = splitmix64(mixed ^ stream);
    return {
        static_cast<std::uint32_t>(mixed),
        static_cast<std::uint32_t>(mixed >> 32)};
}

std::array<double, 4> gaussian_group(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t direction,
    std::uint64_t group) {
    // One counter names four rows of one random direction.
    std::array<std::uint32_t, 4> counter = {
        static_cast<std::uint32_t>(group),
        static_cast<std::uint32_t>(group >> 32),
        static_cast<std::uint32_t>(direction),
        static_cast<std::uint32_t>(direction >> 32)};

    std::array<std::uint32_t, 4> bits =
        philox4x32(counter, make_key(seed, stream));

    double u_0 = (static_cast<double>(bits[0]) + 0.5) * uint32_scale;
    double u_1 = (static_cast<double>(bits[1]) + 0.5) * uint32_scale;
    double u_2 = (static_cast<double>(bits[2]) + 0.5) * uint32_scale;
    double u_3 = (static_cast<double>(bits[3]) + 0.5) * uint32_scale;

    double radius_0 = std::sqrt(-2.0 * std::log(u_0));
    double angle_0 = two_pi * u_1;
    double radius_1 = std::sqrt(-2.0 * std::log(u_2));
    double angle_1 = two_pi * u_3;

    // Two Box-Muller pairs give four independent normal coordinates.
    return {
        radius_0 * std::cos(angle_0),
        radius_0 * std::sin(angle_0),
        radius_1 * std::cos(angle_1),
        radius_1 * std::sin(angle_1)};
}

}  // namespace

double gaussian_value(
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t direction,
    std::uint64_t row) {
    std::array<double, 4> values =
        gaussian_group(seed, stream, direction, row / 4);
    return values[row % 4];
}

void fill_gaussian(
    double* values,
    int rows,
    int cols,
    std::uint64_t seed,
    std::uint64_t stream,
    std::uint64_t first_direction) {
    for (int col = 0; col < cols; ++col) {
        std::uint64_t direction = first_direction + col;
        for (int first_row = 0; first_row < rows; first_row += 4) {
            std::array<double, 4> group = gaussian_group(
                seed,
                stream,
                direction,
                static_cast<std::uint64_t>(first_row / 4));

            int group_size = rows - first_row;
            if (group_size > 4) {
                group_size = 4;
            }
            for (int offset = 0; offset < group_size; ++offset) {
                values[first_row + offset + col * rows] = group[offset];
            }
        }
    }
}

}  // namespace ac_rsvd::math
