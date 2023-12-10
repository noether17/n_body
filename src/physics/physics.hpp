#pragma once

#include <span>
#include <vector>

#include "Vector3d.hpp"

constexpr auto G = 6.67e-11; // gravitational constant
constexpr auto L = 1.0; // box width
constexpr auto epsilon = 1e-3*L; // softening parameter
constexpr auto softening2 = epsilon*epsilon;

constexpr auto characteristic_time(int N, double L) -> double
{
    return std::sqrt(L*L*L / (G * N));
}

struct State
{
    std::vector<Vector3d> pos{};
    std::vector<Vector3d> vel{};
};

// Assumes that this is called from a thread computing a portion of the accelerations,
// identified by the offset and the size of the portion.
void threaded_gravity(const std::span<const Vector3d> pos, std::size_t offset, std::span<Vector3d> acc)
{
    for (auto acc_index = offset; acc_index < offset + acc.size(); ++acc_index)
    {
        // acc_index is the index among the full system of the particle whose acceleration is being
        // computed.
        auto& current_acc = acc[acc_index - offset];
        current_acc = Vector3d{};
        for (std::size_t pos_index = 0; pos_index < pos.size(); ++pos_index)
        {
            if (acc_index == pos_index) { continue; }
            auto r = pos[pos_index] - pos[acc_index];
            auto r_mag2 = mag2(r);
            auto denominator = (r_mag2 + softening2) * std::sqrt(r_mag2);
            current_acc += r / denominator;
        }
        current_acc *= G;
    }
}