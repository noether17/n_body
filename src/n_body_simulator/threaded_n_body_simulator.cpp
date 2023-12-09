#include <random>
#include <vector>

#include "ode.hpp"
#include "physics.hpp"
#include "Vector3d.hpp"

constexpr auto N = 1 << 3; // number of particles

int main()
{
    // initialize state
    auto state = State{};
    state.pos = std::vector<Vector3d>(N);
    state.vel = std::vector<Vector3d>(N);
    auto seed = 25UL;
    auto gen = std::mt19937{seed};
    auto dist = std::uniform_real_distribution<double>(0.0, L);
    for (auto& pos : state.pos)
    {
        pos = Vector3d{dist(gen), dist(gen), dist(gen)};
    }

    // simulate
    auto max_time = characteristic_time(N, L);
    auto dt = 1e-1*max_time;
    threaded_euler(state, dt, max_time, 4);

    return 0;
}