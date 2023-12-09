#pragma once

#include <barrier>
#include <span>
#include <thread>
#include <vector>

#include "physics.hpp"
#include "Vector3d.hpp"

struct ThreadData
{
    std::span<Vector3d> pos{}; // full system positions
    std::span<Vector3d> vel{}; // thread portion of velocities
    std::span<Vector3d> acc{}; // thread portion of accelerations
    double dt{0.0};
    std::size_t offset{0}; // offset of thread portion within full system
};

template <typename AccFunc, typename BarrierType>
void threaded_euler_step(ThreadData& data, AccFunc acc_func,
    BarrierType& barrier)
{
    barrier.arrive_and_wait();
    acc_func(data.pos, data.offset, data.acc);
    barrier.arrive_and_wait();
    auto thread_size = data.vel.size();
    for (auto i = 0; i < thread_size; ++i) { data.pos[i + data.offset] += data.vel[i]*data.dt; }
    for (auto i = 0; i < thread_size; ++i) { data.vel[i] += data.acc[i]*data.dt; }
}

template <typename AccFunc, typename BarrierType>
void threaded_euler_loop(ThreadData data, double max_time, AccFunc acc_func,
    BarrierType& barrier)
{
    auto t = 0.0;
    while (t < max_time)
    {
        threaded_euler_step(data, acc_func, barrier);
        t += data.dt;
    }
}

void threaded_euler(State& state, double dt, double max_time, int n_threads)
{
    auto pos = std::span<Vector3d>{state.pos};
    auto thread_size = pos.size() / n_threads;
    auto acc_portions = std::vector(n_threads, std::vector<Vector3d>(thread_size));
    auto barrier = std::barrier{n_threads};
    auto threads = std::vector<std::jthread>{};
    for (auto i = 0; i < n_threads; ++i)
    {
        auto offset = i*thread_size;
        auto vel_portion = std::span<Vector3d>{state.vel}.subspan(offset, thread_size);
        auto acc_portion = std::span<Vector3d>{acc_portions[i]};
        auto data = ThreadData{pos, vel_portion, acc_portion, dt, offset};
        auto thread_function = [&](){ threaded_euler_loop(data, max_time, threaded_gravity, barrier); };
        threads.emplace_back(thread_function);
    }
}