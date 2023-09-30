#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

constexpr auto N = 1 << 10; // number of particles
constexpr auto dim = 3; // number of spatial dimensions
constexpr auto L = 10.0; // box length

struct State
{
    std::vector<double> pos{};
    std::vector<double> vel{};
};

void output_states(double t, const State& state, const std::string& fn);

int main()
{
    // initialize states
    auto state = State{};
    state.pos = std::vector<double>(dim*N);
    state.vel = std::vector<double>(dim*N);
    auto rd = std::random_device{};
    auto gen = std::mt19937{rd()};
    auto dist = std::uniform_real_distribution<double>(L);
    for (auto& pos : state.pos) { pos = dist(gen); }

    // print states
    output_states(0.0, state, "output.txt");

    // rotate around z-axis
    constexpr auto theta = 2.0 * 3.14159265358979323 / 100.0;
    auto cos_theta = std::cos(theta);
    auto sin_theta = std::sin(theta);
    for (auto t_idx = 1; t_idx <= 100; ++t_idx)
    {
        auto t = t_idx * 0.1;
        for (auto i = 0; i < N; ++i)
        {
            auto x = state.pos[3*i];
            auto y = state.pos[3*i + 1];
            state.pos[3*i] = cos_theta*x - sin_theta*y;
            state.pos[3*i + 1] = sin_theta*x + cos_theta*y;
        }
        output_states(t, state, "output.txt");
    }

    return 0;
}

void output_states(double t, const State& state, const std::string& fn)
{
    auto output_file = std::ofstream{fn, std::ios_base::app};
    output_file << std::setprecision(16);
    for (auto i = 0; i != state.pos.size() / dim; ++i)
    {
        output_file << t << ','
                    << state.pos[i*3] << ','
                    << state.pos[i*3 + 1] << ','
                    << state.pos[i*3 + 2] << ','
                    << state.vel[i*3] << ','
                    << state.vel[i*3 + 1] << ','
                    << state.vel[i*3 + 2] << '\n';
    }
}