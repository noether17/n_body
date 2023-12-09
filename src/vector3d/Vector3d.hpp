#pragma once

#include <cmath>

struct Vector3d
{
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

constexpr auto operator+=(Vector3d& v, const Vector3d& w) -> Vector3d&
{
    v.x += w.x;
    v.y += w.y;
    v.z += w.z;
    return v;
}

constexpr auto operator-=(Vector3d& v, const Vector3d& w) -> Vector3d&
{
    v.x -= w.x;
    v.y -= w.y;
    v.z -= w.z;
    return v;
}

constexpr auto operator*=(Vector3d& v, double s) -> Vector3d&
{
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}

constexpr auto operator/=(Vector3d& v, double s) -> Vector3d&
{
    auto recip = 1.0 / s;
    v.x *= recip;
    v.y *= recip;
    v.z *= recip;
    return v;
}

constexpr auto operator+(const Vector3d& v, const Vector3d& w) -> Vector3d
{
    auto u = v;
    return u += w;
}

constexpr auto operator-(const Vector3d& v, const Vector3d& w) -> Vector3d
{
    auto u = v;
    return u -= w;
}

constexpr auto operator*(const Vector3d& v, double s) -> Vector3d
{
    auto u = v;
    return u *= s;
}

constexpr auto operator/(const Vector3d& v, double s) -> Vector3d
{
    auto u = v;
    return u /= s;
}

constexpr auto operator*(double s, const Vector3d& v) -> Vector3d
{
    return v*s;
}

constexpr auto mag2(const Vector3d& v) -> double
{
    return v.x*v.x + v.y*v.y + v.z*v.z;
}

constexpr auto mag(const Vector3d& v) -> double
{
    return std::sqrt(mag2(v));
}