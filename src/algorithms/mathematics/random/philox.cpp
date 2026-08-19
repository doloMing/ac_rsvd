#include "algorithms/mathematics/random/philox.hpp"

#include <cstdint>

namespace ac_rsvd::math {
namespace {

constexpr std::uint32_t multiplier_0 = 0xd2511f53U;
constexpr std::uint32_t multiplier_1 = 0xcd9e8d57U;
constexpr std::uint32_t key_step_0 = 0x9e3779b9U;
constexpr std::uint32_t key_step_1 = 0xbb67ae85U;

std::array<std::uint32_t, 4> philox_round(
    const std::array<std::uint32_t, 4>& counter,
    const std::array<std::uint32_t, 2>& key) {
    std::uint64_t product_0 =
        static_cast<std::uint64_t>(multiplier_0) * counter[0];
    std::uint64_t product_1 =
        static_cast<std::uint64_t>(multiplier_1) * counter[2];

    std::uint32_t low_0 = static_cast<std::uint32_t>(product_0);
    std::uint32_t high_0 = static_cast<std::uint32_t>(product_0 >> 32);
    std::uint32_t low_1 = static_cast<std::uint32_t>(product_1);
    std::uint32_t high_1 = static_cast<std::uint32_t>(product_1 >> 32);

    return {
        high_1 ^ counter[1] ^ key[0],
        low_1,
        high_0 ^ counter[3] ^ key[1],
        low_0};
}

}  // namespace

std::array<std::uint32_t, 4> philox4x32(
    std::array<std::uint32_t, 4> counter,
    std::array<std::uint32_t, 2> key) {
    for (int round = 0; round < 10; ++round) {
        counter = philox_round(counter, key);
        key[0] += key_step_0;
        key[1] += key_step_1;
    }
    return counter;
}

}  // namespace ac_rsvd::math
