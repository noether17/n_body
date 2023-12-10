#pragma once

#include <algorithm>
#include <barrier>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <span>
#include <sstream>
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
    for (std::size_t i = 0; i < thread_size; ++i) { data.pos[i + data.offset] += data.vel[i]*data.dt; }
    for (std::size_t i = 0; i < thread_size; ++i) { data.vel[i] += data.acc[i]*data.dt; }
}

struct OutputEntry
{
    double t{0.0};
    std::size_t index{0};
    Vector3d pos{};
    Vector3d vel{};
};

template <typename AccFunc, typename BarrierType>
void threaded_euler_loop(ThreadData data, double max_time, AccFunc acc_func,
    BarrierType& barrier, std::mutex& mx, std::vector<OutputEntry>& out_states)
{
    auto t = 0.0;
    auto local_out_states = out_states;
    local_out_states.reserve(data.vel.size()*max_time/data.dt);
    while (t < max_time)
    {
        threaded_euler_step(data, acc_func, barrier);
        t += data.dt;

        for (std::size_t i = 0; i < data.vel.size(); ++i)
        {
            auto pos_index = i + data.offset;
            local_out_states.emplace_back(t, pos_index, data.pos[pos_index], data.vel[i]);
        }
    }

    std::scoped_lock lock{mx};
    out_states = std::move(local_out_states);
}

auto generate_filename(int system_size, int n_threads) -> std::string
{
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto filename = std::stringstream{};
    filename << "output_" << system_size << '_' << n_threads << '_' << now_c << ".csv";
    return filename.str();
}

auto threaded_euler(State &state, double dt, double max_time, int n_threads)
{
    auto pos = std::span<Vector3d>{state.pos};
    auto system_size = pos.size();
    auto thread_size = system_size / n_threads;
    auto acc_portions = std::vector(n_threads, std::vector<Vector3d>(thread_size));
    auto barrier = std::barrier{n_threads};
    auto threads = std::vector<std::jthread>{};
    auto thread_output_vecs = std::vector(n_threads, std::vector<OutputEntry>{});
    std::mutex mx;
    for (auto i = 0; i < n_threads; ++i)
    {
        auto offset = i*thread_size;
        auto vel_portion = std::span<Vector3d>{state.vel}.subspan(offset, thread_size);
        auto acc_portion = std::span<Vector3d>{acc_portions[i]};
        auto data = ThreadData{pos, vel_portion, acc_portion, dt, offset};
        auto& thread_output = thread_output_vecs[i];
        auto thread_function = [&barrier, &mx, &thread_output]
            (ThreadData data, double max_time, auto acc_func)
        {
            threaded_euler_loop(data, max_time, threaded_gravity, barrier, mx, thread_output);
        };
        threads.emplace_back(thread_function, data, max_time, threaded_gravity);
    }
    for (auto& thread : threads) { thread.join(); }

    return thread_output_vecs;
}

void output_results(std::size_t system_size, int n_threads,
    std::vector<std::vector<OutputEntry>>& thread_output_vecs)
{
    auto output_vector = std::vector<OutputEntry>{};
    for (auto& thread_output : thread_output_vecs)
    {
        output_vector.insert(output_vector.end(), thread_output.begin(), thread_output.end());
    }
    std::sort(output_vector.begin(), output_vector.end(),
        [](const auto& a, const auto& b) { return a.index < b.index; });
    std::stable_sort(output_vector.begin(), output_vector.end(),
        [](const auto& a, const auto& b) { return a.t < b.t; });

    auto filename = generate_filename(system_size, n_threads);
    auto out = std::ofstream{filename};
    for (const auto& entry : output_vector)
    {
        out << entry.t << ','
            << entry.index << ','
            << entry.pos.x << ','
            << entry.pos.y << ','
            << entry.pos.z << ','
            << entry.vel.x << ','
            << entry.vel.y << ','
            << entry.vel.z << '\n';
    }
}