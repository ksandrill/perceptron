//
// Created by azari on 18.04.2021.
//

#ifndef PERCEPTRON_CPP_UTIL_H
#define PERCEPTRON_CPP_UTIL_H
#include <cmath>
template<typename T>
constexpr bool is_float_or_double =
        std::is_same<T, float>::value || std::is_same<T, double>::value;

template<typename T>
inline auto sigmoid(T x) -> T {
    static_assert(is_float_or_double<T>);
    return 1 / (1 + exp(-x));
}


template<typename T, size_t size>
auto MSE(const Vector<T, size> &errorVector) -> T {
    static_assert(is_float_or_double<T>);
    T MSE_error = 0;
    for (auto i = 0; i < size; ++i) {
        MSE_error += std::pow(errorVector[i], 2);

    }
    return MSE_error;

}


template<typename T>
inline auto derivative_sigmoid(T sigmoid_value) -> T {
    static_assert(is_float_or_double<T>);
    return sigmoid_value * (1 - sigmoid_value);
}


#endif //PERCEPTRON_CPP_UTIL_H
