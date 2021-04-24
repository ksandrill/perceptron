//
// Created by azari on 25.04.2021.
//

#ifndef TRY_AGAIN_PERCEPTRON_LAYER_H
#define TRY_AGAIN_PERCEPTRON_LAYER_H

#include "./NumCpp.hpp"
#include "util.h"
using NdArrayF = nc::NdArray<float>;

class Layer {
private:
    NdArrayF layerOutput;
    NdArrayF layerWeights;

public:
    Layer(unsigned layerSize, unsigned prevLayerSize) {
        layerOutput = nc::zeros<float>({1, layerSize});
        layerWeights = nc::zeros<float>({prevLayerSize, layerSize});

    };

    void initWeights(float low, float high) {
        layerWeights = nc::random::uniform<float>(nc::shape(layerWeights), low, high);


    }

    void activateNeurones(const NdArrayF &input){
        layerOutput = nc::matmul(input,layerWeights);
        auto outputSize = layerOutput.size();
        for (auto i = 0; i < outputSize; ++i) {
            float neuroneActivation = sigmoid(layerOutput[i]);
            layerOutput[i] = std::isnan(neuroneActivation) ? 0.f : neuroneActivation;

        }


    }

    const NdArrayF &getLayerOutput() const {
        return layerOutput;
    }

    void setLayerOutput(const NdArrayF &layerOutput) {
        Layer::layerOutput = layerOutput;
    }

    const NdArrayF &getLayerWeights() const {
        return layerWeights;
    }

    void setLayerWeights(const NdArrayF &layerWeights) {
        Layer::layerWeights = layerWeights;
    }

};


#endif //TRY_AGAIN_PERCEPTRON_LAYER_H
