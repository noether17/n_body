#include <random>
#include <vector>

#include "ode.hpp"
#include "physics.hpp"
#include "Vector3d.hpp"

auto N = 1 << 3; // number of particles
auto n_threads = 1 << 2; // number of threads

int main(int argc, char** argv)
{
    if (argc > 1) { N = std::stoi(argv[1]); }
    if (argc > 2) { n_threads = std::stoi(argv[2]); }

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
    auto dt = 1e-3*max_time;
    threaded_euler(state, dt, max_time, n_threads);

    return 0;
}