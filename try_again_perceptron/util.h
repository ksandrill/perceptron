//
// Created by azari on 25.04.2021.
//

#ifndef TRY_AGAIN_PERCEPTRON_UTIL_H
#define TRY_AGAIN_PERCEPTRON_UTIL_H

inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

inline float mse(const nc::NdArray<float> &error) {
    auto mse = 0.f;
    auto len = error.size();
    for (auto i = 0; i < len; ++i) {
        mse += (error[i] * error[i]);
    }
    mse/= len;
    return mse;
}

inline float rMse(const nc::NdArray<float> &error){
    return nc::sqrt(mse(error));
}
#endif //TRY_AGAIN_PERCEPTRON_UTIL_H
