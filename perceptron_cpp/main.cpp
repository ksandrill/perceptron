#include <iostream>
#include "math/Vector.h"
#include "math/Mat.h"
#include "Perceptron.h"

namespace {
    constexpr auto epoch_count = 5000;
}

auto main() -> int {
    ////Perceptron<Type, input_neurones,output_neurones,first_neurones,second neurones>
    perceptronStaff::Perceptron<float, 2, 1, 2, 2> perceptron{};
    perceptron.initWeights(-1, 1);
    Mat<float, 4, 2> input{{0, 0},
                           {0, 1},
                           {1, 0},
                           {1, 1}};
    Vector<Vector<float, 1>, 4> output{{0},
                                       {1},
                                       {1},
                                       {1}};
    for (auto epoch = 0; epoch < epoch_count; ++epoch) {
        for (auto i = 0; i < 4; ++i) {
            perceptron.setInputLayer(input[i]);
            perceptron.feedForward();
            perceptron.backProp(output[i]);
        }

    }
    for (auto i = 0; i < 4; ++i) {
        perceptron.setInputLayer(input[i]);
        std::cout << "input: " << perceptron.getInputLayer();
        perceptron.feedForward();
        std::cout << " output: " << perceptron.getOutputLayerOutput() << std::endl;


    }

    return 0;
}

