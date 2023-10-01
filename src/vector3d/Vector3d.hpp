class Vector3d
{
    public:
    Vector3d(double x = 0.0, double y = 0.0, double z = 0.0)
        : x_{x}, y_{y}, z_{z} {}

    Vector3d(const Vector3d& v) = default;
    auto operator=(const Vector3d& v) -> Vector3d& = default;
    Vector3d(Vector3d&& v) = default;
    auto operator=(Vector3d&& v) -> Vector3d& = default;
    ~Vector3d() = default;

    auto get_x() const -> double { return x_; }
    auto get_y() const -> double { return y_; }
    auto get_z() const -> double { return z_; }

    auto operator+=(const Vector3d& v) -> Vector3d&
    {
        x_ += v.x_;
        y_ += v.y_;
        z_ += v.z_;
        return *this;
    }

    auto operator-=(const Vector3d& v) -> Vector3d&
    {
        x_ -= v.x_;
        y_ -= v.y_;
        z_ -= v.z_;
        return *this;
    }

    auto operator*=(double s) -> Vector3d&
    {
        x_ *= s;
        y_ *= s;
        z_ *= s;
        return *this;
    }

    auto operator+(const Vector3d& v) const -> Vector3d
    {
        auto w = *this;
        return w += v;
    }

    auto operator-(const Vector3d& v) const -> Vector3d
    {
        auto w = *this;
        return w -= v;
    }

    auto operator*(double s) const -> Vector3d
    {
        auto w = *this;
        return w *= s;
    }

    private:
    double x_{};
    double y_{};
    double z_{};
};

auto operator*(double s, const Vector3d& v) -> Vector3d
{
    return v*s;
}