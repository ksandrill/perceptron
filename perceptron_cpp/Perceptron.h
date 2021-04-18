//
// Created by azari on 16.04.2021.
//

#ifndef PERCEPTRON_CPP_PERCEPTRON_H
#define PERCEPTRON_CPP_PERCEPTRON_H

#include <cstddef>
#include "math/Vector.h"
#include "math/Mat.h"
#include "math/Util.h"
#include "random"

namespace perceptronStaff {
    constexpr auto ETA = 0.3;

    template<typename T, size_t prev_size, size_t cur_layer_size>
    auto feedForwardStep(const Vector<T, prev_size> &input,
                         const Mat<T, cur_layer_size, prev_size> &curWeights) -> Vector<T, cur_layer_size> {
        Vector<T, cur_layer_size> neuroneOutput{};
        for (auto neurone = 0; neurone < cur_layer_size; ++neurone) {
            neuroneOutput[neurone] = sigmoid(scalar_product(input, curWeights[neurone]));
        }
        return neuroneOutput;

    }

    template<typename T, size_t output_layer_size>
    auto calcOutputDelta(const Vector<T, output_layer_size> &errorVector,
                         const Vector<T, output_layer_size> outputLayerOutput) -> Vector<T, output_layer_size> {
        Vector<T, output_layer_size> delta{};
        for (auto i = 0; i < output_layer_size; ++i) {
            delta[i] = errorVector[i] * derivative_sigmoid(outputLayerOutput[i]);
        }
        return delta;


    }

    template<typename T, size_t cur_layer_size, size_t prev_layer_size>
    auto calcLayerWeightCorr(const Vector<T, cur_layer_size> &delta,
                             const Vector<T, prev_layer_size> &prevOutput) -> Mat<T, cur_layer_size, prev_layer_size> {
        Mat<T, cur_layer_size, prev_layer_size> weightCorr{};
        for (auto neurone = 0; neurone < cur_layer_size; ++neurone) {
            for (auto weight = 0; weight < prev_layer_size; ++weight) {
                weightCorr[neurone][weight] = -ETA * delta[neurone] * prevOutput[weight];
            }
        }
        return weightCorr;


    }


    template<typename T, size_t cur_layer_size, size_t next_layer_size>
    auto
    calcHiddenLayerDelta(const Vector<T, cur_layer_size> &curOutput, const Vector<T, next_layer_size> &nextLayerDelta,
                         const Mat<T, next_layer_size, cur_layer_size> &nextLayerWeights) -> Vector<T, cur_layer_size> {
        Vector<T, cur_layer_size> hiddenLayerDelta{};
        for (auto k = 0; k < cur_layer_size; ++k) {
            T sum = 0.0;
            for (auto j = 0; j < next_layer_size; ++j) {
                sum += nextLayerWeights[j][k] * nextLayerDelta[j];
            }
            hiddenLayerDelta[k] = derivative_sigmoid(curOutput[k]) * sum;
        }
        return hiddenLayerDelta;

    }


    template<typename T, size_t input_layer_size, size_t output_layer_size, size_t first_layer_size, size_t second_layer_size>
    class Perceptron {
        static_assert(is_float_or_double<T>);

    private:
        Vector<T, input_layer_size> inputLayer{};
        ///first layer
        Mat<T, first_layer_size, input_layer_size> firstLayerWeights{};
        Vector<T, first_layer_size> firstLayerOutput{};
        ///second layer
        Mat<T, second_layer_size, first_layer_size> secondLayerWeights{};
        Vector<T, second_layer_size> secondLayerOutput{};
        ///output layer
        Mat<T, output_layer_size, second_layer_size> outputLayerWeights{};
        Vector<T, output_layer_size> outputLayerOutput{};


    public:
        void initWeights(T a, T b) {
            static_assert(is_float_or_double<T>);
            auto distribution = std::uniform_real_distribution<T>(a, b);
            auto rd = std::random_device();
            auto gen = std::mt19937(rd());
            for (auto i = 0; i < first_layer_size; ++i) {
                for (auto j = 0; j < input_layer_size; ++j) {
                    firstLayerWeights[i][j] = distribution(gen);
                }
            }

            for (auto i = 0; i < second_layer_size; ++i) {
                for (auto j = 0; j < first_layer_size; ++j) {
                    secondLayerWeights[i][j] = distribution(gen);
                }
            }

            for (auto i = 0; i < output_layer_size; ++i) {
                for (auto j = 0; j < second_layer_size; ++j) {
                    outputLayerWeights[i][j] = distribution(gen);
                }
            }

        }

        void backProp(const Vector<T, output_layer_size> &target) {
            auto error = outputLayerOutput - target;
            auto outputDeltaVector = calcOutputDelta(error, outputLayerOutput);
            auto outputWeightCor = calcLayerWeightCorr(outputDeltaVector, secondLayerOutput);
            auto secondDelta = calcHiddenLayerDelta(secondLayerOutput, outputDeltaVector, outputLayerWeights);
            auto secondWeightCor = calcLayerWeightCorr(secondDelta, firstLayerOutput);
            auto firstDelta = calcHiddenLayerDelta(firstLayerOutput, secondDelta, secondLayerWeights);
            auto firstWeightCor = calcLayerWeightCorr(firstDelta, inputLayer);
            firstLayerWeights += firstWeightCor;
            secondLayerWeights += secondWeightCor;
            outputLayerWeights += outputWeightCor;
        }

        void feedForward() {
            firstLayerOutput = feedForwardStep(inputLayer, firstLayerWeights);
            secondLayerOutput = feedForwardStep(firstLayerOutput, secondLayerWeights);
            outputLayerOutput = feedForwardStep(secondLayerOutput, outputLayerWeights);


        }


        const Vector<T, input_layer_size> &getInputLayer() const {
            return inputLayer;
        }


        const Vector<T, output_layer_size> &getOutputLayerOutput() const {
            return outputLayerOutput;
        }


        void setInputLayer(const Vector<T, input_layer_size> &inputLayer_) {
            Perceptron::inputLayer = inputLayer_;
        }


    };
}


#endif //PERCEPTRON_CPP_PERCEPTRON_H
