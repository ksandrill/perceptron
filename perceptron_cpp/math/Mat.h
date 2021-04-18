//
// Created by azari on 17.04.2021.
//

#ifndef PERCEPTRON_CPP_MAT_H
#define PERCEPTRON_CPP_MAT_H

#include "Vector.h"
#include "Util.h"



template<typename T, size_t ROWS, size_t COLUMNS>
using Mat = Vector<Vector<T, COLUMNS>, ROWS>;


template<typename T, size_t ROWS, size_t COLUMNS>
auto operator<<(std::ostream &out, const Mat<T, ROWS, COLUMNS> &mat) -> std::ostream & {
    out << "[" << std::endl;
    auto last_elem_index = ROWS - 1;
    for (auto i = 0; i < last_elem_index; ++i) {
        out << " " << mat[i] << "," << std::endl;
    }
    out << " " << mat[last_elem_index] << std::endl;
    out << "]" << std::endl;
    return out;
}

template<typename T, size_t N>
auto scalar_product(const Vector<T, N> &v1, const Vector<T, N> &v2) -> T {
    static_assert(is_float_or_double < T > );
    T dot = 0;
    for (auto i = 0; i < N; ++i) {
        dot += v1[i] * v2[i];
    }

    return dot;


}





#endif //PERCEPTRON_CPP_MAT_H
