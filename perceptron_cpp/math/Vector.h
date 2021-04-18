//
// Created by azari on 16.04.2021.
//

#ifndef PERCEPTRON_CPP_VECTOR_H
#define PERCEPTRON_CPP_VECTOR_H

#include <array>
#include <stdexcept>


#include <cstddef>
#include "iostream"

template<typename T, size_t N>
class Vector : private std::array<T, N> {
public:
    using std::array<T, N>::operator[];
    explicit Vector(T item) {
        fill(item);
    };

    Vector() = default;

    Vector(Vector const &) = default;

    Vector(Vector &&) noexcept = default;

    ~Vector() = default;

    auto operator=(const Vector &other) -> Vector & = default;

    Vector(std::initializer_list<T> const &initializerList) {
        auto i = 0;
        for (auto value:initializerList) {
            (*this)[i] = value;
            ++i;
        }
    }



///one arg operators
    auto operator*=(const T &val) -> Vector {
        auto size = (*this).size();
        for (auto i = 0; i < size; ++i) {
            (*this)[i] *= val;
        }
        return *this;
    }

    auto operator/=(const T &val) -> Vector {
        auto size = (*this).size();
        for (auto i = 0; i < size; ++i) {
            (*this)[i] /= val;
        }
        return *this;
    }


    auto operator+=(const Vector &other) {
        auto size = (*this).size();
        for (auto i = 0; i < size; ++i) {
            (*this)[i] += other[i];
        }
        return *this;
    }

    auto operator-=(const Vector &other) -> Vector {
        auto size = (*this).size();
        for (auto i = 0; i < size; ++i) {
            (*this)[i] -= other[i];
        }
        return *this;
    }
///binary operators

    auto operator/(const T &val) {
        return Vector(*this) /= val;
    }

    auto operator*(const T &val) {
        return Vector(*this) *= val;
    }

    auto operator+(const Vector &other) {
        return Vector(*this) += other;
    }

    auto operator-(const Vector &other) {
        return Vector(*this) -= other;
    }

///output operator



};

template<typename T, size_t N>
auto operator<<(std::ostream &out, const Vector<T, N> &vector) -> std::ostream & {
    out << "[";
    auto last_elem_index = N - 1;
    for (auto i = 0; i < last_elem_index; ++i) {
        out << vector[i] << ",";
    }
    out << vector[last_elem_index];
    out << "]";
    return out;
}



#endif //PERCEPTRON_CPP_VECTOR_H
