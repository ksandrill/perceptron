//
// Created by azari on 25.04.2021.
//

#ifndef TRY_AGAIN_PERCEPTRON_PERCEPTRON_H
#define TRY_AGAIN_PERCEPTRON_PERCEPTRON_H

#include "Layer.h"

using Data = std::pair<NdArrayF, NdArrayF>;
using DataSet = std::vector<Data>;

class Perceptron {
private:
    NdArrayF inputLayer;
    Layer firstLayer;
    Layer secondLayer;
    Layer outputLayer;

    NdArrayF calcOutputDelta(const NdArrayF &error) {
        return outputLayer.getLayerOutput() * (1.f - outputLayer.getLayerOutput()) * error;


    }

    NdArrayF calcHiddenDelta(const NdArrayF &nextLayerDelta, const NdArrayF &nextLayerInput,
                             const NdArrayF &nextLayerWeights) {
        auto curDelta = nc::matmul(nextLayerDelta, nextLayerWeights.transpose());
        return nextLayerInput * (1.f - nextLayerInput) * curDelta;

    }

    NdArrayF calcWeightCorr(const NdArrayF &layerDelta, const NdArrayF &layerInput) {
        return nc::matmul(layerInput.transpose(), layerDelta);

    }


public:
    Perceptron(unsigned inputLayerSize, unsigned firstLayerSize, unsigned secondLayerSize, unsigned outputLayerSize) :
            firstLayer(firstLayerSize, inputLayerSize), secondLayer(secondLayerSize, firstLayerSize),
            outputLayer(outputLayerSize, secondLayerSize) {
        inputLayer = nc::zeros<float>({1, inputLayerSize});

    }


    void initWeights(float low, float high) {
        firstLayer.initWeights(low, high);
        secondLayer.initWeights(low, high);
        outputLayer.initWeights(low, high);

    }

    void feedForward(const NdArrayF &input) {
        inputLayer = input;
        firstLayer.activateNeurones(inputLayer);
        secondLayer.activateNeurones(firstLayer.getLayerOutput());
        outputLayer.activateNeurones(secondLayer.getLayerOutput());


    }


    void backProp(const NdArrayF &error, float lr) {
        auto outputLayerDelta = calcOutputDelta(error);
        auto outputLayerWeightCorr = calcWeightCorr(outputLayerDelta, secondLayer.getLayerOutput());
        auto secondLayerDelta = calcHiddenDelta(outputLayerDelta, secondLayer.getLayerOutput(),
                                                outputLayer.getLayerWeights());
        auto secondLayerWeightCorr = calcWeightCorr(secondLayerDelta, firstLayer.getLayerOutput());
        auto firstLayerDelta = calcHiddenDelta(secondLayerDelta, firstLayer.getLayerOutput(),
                                               secondLayer.getLayerWeights());
        auto firstLayerWeightCorr = calcWeightCorr(firstLayerDelta, inputLayer);
        firstLayer.setLayerWeights(firstLayer.getLayerWeights() - lr * firstLayerWeightCorr);
        secondLayer.setLayerWeights(secondLayer.getLayerWeights() - lr * secondLayerWeightCorr);
        outputLayer.setLayerOutput(outputLayer.getLayerWeights() - lr * outputLayerWeightCorr);
    }

    NdArrayF train(const DataSet &dataSet, unsigned epochCount, float lr) {
        auto iterations = dataSet.size();
        auto mseVector = nc::zeros<float>({1, epochCount});
        for (auto epoch = 0; epoch < epochCount; ++epoch) {
            std::cout << "epoch: " << epoch << std::endl;
            auto avgMse = 0.f;
            for (auto i = 0; i < iterations; ++i) {
                feedForward(dataSet[i].first);
                auto error = outputLayer.getLayerOutput() - dataSet[i].second;
                backProp(error, lr);
                avgMse += mse(error);

            }
            mseVector[epoch] = avgMse/iterations;


        }
        return mseVector;
    }


    DataSet getTestResult(const DataSet &dataSet) {
        auto iterations = dataSet.size();
        DataSet res{};
        auto inputSize = inputLayer.size();
        auto outputSize = outputLayer.getLayerOutput().size();
        for (auto i = 0; i < iterations; ++i) {
            std::cout << "input: " << dataSet[i].first << " ";
            feedForward(dataSet[i].first);
            std::cout << "output: " << outputLayer.getLayerOutput() << " ";
            std::cout << "real output: " << dataSet[i].second << std::endl;
            auto aux = Data();
            aux.first = nc::zeros<float>({1, inputSize});
            aux.first = dataSet[i].first;
            aux.second = nc::zeros<float>({1, outputSize});
            aux.second = outputLayer.getLayerOutput();
            res.emplace_back(aux);
        }
        return res;


    }

};


#endif //TRY_AGAIN_PERCEPTRON_PERCEPTRON_H
