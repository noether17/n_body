#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

#include "Vector3d.hpp"

constexpr auto N = 1 << 10; // number of particles
constexpr auto L = 10.0; // box length

struct State
{
    std::vector<Vector3d> pos{};
    std::vector<Vector3d> vel{};
};

void output_states(double t, const State& state, const std::string& fn);

template <typename AccFunc>
void euler_step(std::vector<Vector3d>& pos, std::vector<Vector3d>& vel,
    std::vector<Vector3d>& acc, double dt, AccFunc acc_func)
{
    acc_func(pos, acc);
    for (auto i = 0; i < pos.size(); ++i) { pos[i] += vel[i]*dt; }
    for (auto i = 0; i < vel.size(); ++i) { vel[i] += acc[i]*dt; }
}

template <typename AccFunc>
void leapfrog_step(std::vector<Vector3d>& pos, std::vector<Vector3d>& vel,
    std::vector<Vector3d>& acc, double dt, AccFunc acc_func)
{
    acc_func(pos, acc);
    for (std::size_t i = 0; i < pos.size(); ++i)
    {
        pos[i] += vel[i]*dt + acc[i]*(0.5*dt*dt);
        vel[i] += acc[i]*(0.5*dt);
    }
    acc_func(pos, acc);
    for (std::size_t i = 0; i < vel.size(); ++i)
    {
        vel[i] += acc[i]*(0.5*dt);
    }
}

void gravity(const std::vector<Vector3d>& pos, std::vector<Vector3d>& acc);

int main()
{
    // initialize states
    auto state = State{};
    state.pos = std::vector<Vector3d>(N);
    state.vel = std::vector<Vector3d>(N);
    auto rd = std::random_device{};
    auto gen = std::mt19937{rd()};
    auto dist = std::uniform_real_distribution<double>(L);
    for (auto& pos : state.pos)
    {
        pos = Vector3d{dist(gen), dist(gen), dist(gen)};
    }

    // print states
    output_states(0.0, state, "output.txt");

    // Euler integration
    auto acc = std::vector<Vector3d>(N);
    constexpr auto dt = 1e2;
    for (auto t_idx = 1; t_idx <= 1000; ++t_idx)
    {
        auto t = t_idx * dt;
        //euler_step(state.pos, state.vel, acc, dt, gravity);
        leapfrog_step(state.pos, state.vel, acc, dt, gravity);
        output_states(t, state, "output.txt");
    }

    return 0;
}

void output_states(double t, const State& state, const std::string& fn)
{
    auto output_file = std::ofstream{fn, std::ios_base::app};
    output_file << std::setprecision(16);
    for (std::size_t i = 0; i != state.pos.size(); ++i)
    {
        output_file << t << ','
                    << state.pos[i].x << ','
                    << state.pos[i].y << ','
                    << state.pos[i].z << ','
                    << state.vel[i].x << ','
                    << state.vel[i].y << ','
                    << state.vel[i].z << '\n';
    }
}

void gravity(const std::vector<Vector3d>& pos, std::vector<Vector3d>& acc)
{
    for (auto& a : acc) { a = Vector3d{}; }
    for (std::size_t i = 0; i < pos.size(); ++i)
    {
        for (auto j = i + 1; j < pos.size(); ++j)
        {
            auto dr = pos[j] - pos[i];
            auto r = mag(dr);
            auto r3 = r*r*r;
            constexpr auto G = 6.67e-11;
            auto factor = G / r3;
            acc[i] += dr*factor;
            acc[j] -= dr*factor;
        }
    }
}