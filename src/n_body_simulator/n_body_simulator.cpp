#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

constexpr auto N = 1 << 8; // number of particles
constexpr auto dim = 3; // number of spatial dimensions
constexpr auto L = 10.0; // box length

struct State
{
    std::vector<double> pos{};
    std::vector<double> vel{};
};

void output_states(double t, const State& state, const std::string& fn);

template <typename AccFunc>
void euler_step(std::vector<double>& pos, std::vector<double>& vel,
    std::vector<double>& acc, double dt, AccFunc acc_func)
{
    acc_func(pos, acc);
    for (auto i = 0; i < pos.size(); ++i) { pos[i] += vel[i]*dt; }
    for (auto i = 0; i < vel.size(); ++i) { vel[i] += acc[i]*dt; }
}

void gravity(const std::vector<double>& pos, std::vector<double>& acc);

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

    // Euler integration
    auto acc = std::vector<double>(dim*N);
    constexpr auto dt = 1e3;
    for (auto t_idx = 1; t_idx <= 100; ++t_idx)
    {
        auto t = t_idx * dt;
        euler_step(state.pos, state.vel, acc, dt, gravity);
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

void gravity(const std::vector<double>& pos, std::vector<double>& acc)
{
    for (auto& a : acc) { a = 0.0; }
    for (auto i = 0; i < pos.size() / dim; ++i)
    {
        for (auto j = i + 1; j < pos.size() / dim; ++j)
        {
            auto dx = pos[dim*j] - pos[dim*i];
            auto dy = pos[dim*j + 1] - pos[dim*i + 1];
            auto dz = pos[dim*j + 2] - pos[dim*i + 2];
            auto r = std::sqrt(dx*dx + dy*dy + dz*dz);
            auto r3 = r*r*r;
            constexpr auto G = 6.67e-11;
            auto factor = G / r3;
            acc[dim*i] += dx*factor;
            acc[dim*i + 1] += dy*factor;
            acc[dim*i + 2] += dz*factor;
            acc[dim*j] -= dx*factor;
            acc[dim*j + 1] -= dy*factor;
            acc[dim*j + 2] -= dz*factor;
        }
    }
}