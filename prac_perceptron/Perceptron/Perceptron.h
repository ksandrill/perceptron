//
// Created by azari on 22.04.2021.
//

#ifndef PRAC_PERCEPTRON_PERCEPTRON_H
#define PRAC_PERCEPTRON_PERCEPTRON_H

#include "Layer.h"




using Data = std::pair<NdArrayF, NdArrayF>;
using DataSet = std::vector<Data>;

class Perceptron {
private:
    NdArrayF inputLayer;
    Layer firstLayer;
    Layer secondLayer;
    Layer outputLayer;


    void feedForward() {
        firstLayer.activateNeurones(inputLayer);
        secondLayer.activateNeurones(firstLayer.getLayerOutput());
        outputLayer.activateNeurones(secondLayer.getLayerOutput());


    }

    void backProp(const NdArrayF &error, float lr) {
        auto outputLayerDelta = calcOutputDelta(error);
        auto outputLayerWeightCorr = calcWeightCorr(outputLayerDelta, secondLayer.getLayerOutput());
        auto secondLayerDelta = calcHiddenLayerDelta(outputLayerDelta, secondLayer.getLayerOutput(),
                                                     outputLayer.getLayerWeights());
        auto secondLayerWeightCorr = calcWeightCorr(secondLayerDelta.transpose(), firstLayer.getLayerOutput());
        auto firstLayerDelta = calcHiddenLayerDelta(secondLayerDelta, firstLayer.getLayerOutput(),
                                                    secondLayer.getLayerWeights());
        auto firstLayerWeightCorr = calcWeightCorr(firstLayerDelta.transpose(), inputLayer);
        outputLayer.setLayerWeights(outputLayer.getLayerWeights() - lr * outputLayerWeightCorr);
        secondLayer.setLayerWeights(secondLayer.getLayerWeights() - lr * secondLayerWeightCorr);
        firstLayer.setLayerWeights(firstLayer.getLayerWeights() - lr * firstLayerWeightCorr);


    }


    NdArrayF calcHiddenLayerDelta(const NdArrayF &nextLayerDelta, const NdArrayF &nextLayerInput,
                                  const NdArrayF &nextLayerWeights) {
        auto curDelta = nc::matmul(nextLayerDelta, nextLayerWeights);
        return nextLayerInput * (1.f - nextLayerInput) * curDelta;

    }


    NdArrayF calcOutputDelta(const NdArrayF &error) {
        return error * outputLayer.getLayerOutput() * (1.f - outputLayer.getLayerOutput());
    }

    NdArrayF calcWeightCorr(const NdArrayF &layerDelta, const NdArrayF &layerInput) {
        return nc::matmul(layerDelta, layerInput);
    }


public:
    Perceptron(unsigned inputSize, unsigned firstLayerSize, unsigned secondLayerSize, unsigned outputLayerSize)
            : inputLayer(inputSize), firstLayer(firstLayerSize, inputSize),
              secondLayer(secondLayerSize, firstLayerSize), outputLayer(outputLayerSize, secondLayerSize) {
        inputLayer = nc::zeros<float>({1, inputSize});
    };

    void initWeights(float low, float high) {
        firstLayer.initWeights(low, high);
        secondLayer.initWeights(low, high);
        outputLayer.initWeights(low, high);
    }

    void train(const DataSet &dataSet, unsigned epochCount, float lr) {
        auto iterations = dataSet.size();
        for (auto epoch = 0; epoch < epochCount; ++epoch) {
            for (auto i = 0; i < iterations; ++i) {
                inputLayer = dataSet[i].first;
                feedForward();
                backProp(outputLayer.getLayerOutput() - dataSet[i].second, lr);
            }

        }
    }


    DataSet getTestResult(const DataSet &dataSet) {
        auto iterations = dataSet.size();
        DataSet res{};
        auto inputSize = inputLayer.size();
        auto outputSize = outputLayer.getLayerOutput().size();
        for (auto i = 0; i < iterations; ++i) {
            inputLayer = dataSet[i].first;
            std::cout << "input: " << inputLayer << " ";
            feedForward();
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

    void print_shapes() {
        std::cout << "input_layer" << std::endl;
        std::cout << inputLayer.shape() << std::endl;
        std::cout << " first_layer" << std::endl;
        std::cout << " output:" << std::endl;
        std::cout << firstLayer.getLayerOutput().shape();
        std::cout << " weights:" << std::endl;
        std::cout << firstLayer.getLayerWeights().shape();
        std::cout << " second_layer" << std::endl;

        std::cout << " output:" << std::endl;
        std::cout << secondLayer.getLayerOutput().shape();
        std::cout << " weights:" << std::endl;
        std::cout << secondLayer.getLayerWeights().shape();
        std::cout << " output_layer" << std::endl;

        std::cout << " output:" << std::endl;
        std::cout << outputLayer.getLayerOutput().shape();
        std::cout << " weights:" << std::endl;
        std::cout << outputLayer.getLayerWeights().shape();


    }


};


#endif //PRAC_PERCEPTRON_PERCEPTRON_H
