//
// Created by azari on 22.04.2021.
//

#ifndef PRAC_PERCEPTRON_LAYER_H
#define PRAC_PERCEPTRON_LAYER_H

#include <iostream>
#include <random>
#include "sigmoid.h"
#include "./NumCpp.hpp"

using NdArrayF = nc::NdArray<float>;

class Layer {
private:
    NdArrayF layerOutput;
    NdArrayF layerWeights;
public:
    Layer(unsigned layerSize, unsigned prevLayerSize) {
        layerOutput = nc::zeros<float>({1, layerSize});
        layerWeights = nc::zeros<float>({layerSize, prevLayerSize});

    }

    void initWeights(float low, float high) {
        layerWeights = nc::random::uniform<float>(nc::shape(layerWeights), low, high);


    }

    void activateNeurones(const NdArrayF &input) {
        layerOutput = nc::matmul(input, nc::transpose(layerWeights));
        auto outputSize = layerOutput.size();
        for (auto i = 0; i < outputSize; ++i) {
            float neuroneActivation = sigmoid(layerOutput[i]);
            layerOutput[i] = std::isnan(neuroneActivation) ? 0.f : neuroneActivation;

        }
    }


    const NdArrayF &getLayerOutput() const {
        return layerOutput;
    }

    const NdArrayF &getLayerWeights() const {
        return layerWeights;
    }


    void setLayerOutput(const NdArrayF &layerOutput) {
        Layer::layerOutput = layerOutput;
    }

    void setLayerWeights(const NdArrayF &layerWeights) {
        Layer::layerWeights = layerWeights;
    }

};


#endif //PRAC_PERCEPTRON_LAYER_H
