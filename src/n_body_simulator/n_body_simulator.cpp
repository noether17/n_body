#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

constexpr auto N = 1 << 10; // number of particles
constexpr auto dof = 6; // degrees of freedom per particle
constexpr auto L = 10.0; // box length

void output_states(double t, const std::vector<double>& states, const std::string& fn);

int main()
{
    // initialize states
    auto state_vector = std::vector<double>(dof*N);
    auto rd = std::random_device{};
    auto gen = std::mt19937{rd()};
    auto dist = std::uniform_real_distribution<double>(L);
    for (auto i = 0; i != state_vector.size() / 2; ++i)
    {
        state_vector[i] = dist(gen);
    }

    // print states
    output_states(0.0, state_vector, "output.txt");

    // rotate around z-axis
    constexpr auto theta = 2.0 * 3.14159265358979323 / 100.0;
    auto cos_theta = std::cos(theta);
    auto sin_theta = std::sin(theta);
    for (auto t_idx = 1; t_idx <= 100; ++t_idx)
    {
        auto t = t_idx * 0.1;
        for (auto i = 0; i < N; ++i)
        {
            auto x = state_vector[3*i];
            auto y = state_vector[3*i + 1];
            state_vector[3*i] = cos_theta*x - sin_theta*y;
            state_vector[3*i + 1] = sin_theta*x + cos_theta*y;
        }
        output_states(t, state_vector, "output.txt");
    }

    return 0;
}

void output_states(double t, const std::vector<double>& states, const std::string& fn)
{
    auto output_file = std::ofstream{fn, std::ios_base::app};
    output_file << std::setprecision(16);
    auto vel_offset = states.size() / 2;
    for (auto i = 0; i != states.size() / dof; ++i)
    {
        output_file << t << ','
                    << states[i*3] << ','
                    << states[i*3 + 1] << ','
                    << states[i*3 + 2] << ','
                    << states[i*3 + vel_offset] << ','
                    << states[i*3 + vel_offset + 1] << ','
                    << states[i*3 + vel_offset + 2] << '\n';
    }
}