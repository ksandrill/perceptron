//
// Created by azari on 22.04.2021.
//

#ifndef PRAC_PERCEPTRON_SIGMOID_H
#define PRAC_PERCEPTRON_SIGMOID_H

#endif //PRAC_PERCEPTRON_SIGMOID_H

#include <cmath>

#include "cmath"

inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

inline float derrSigVal(float sig) {
    return sig * (1.f - sig);
}