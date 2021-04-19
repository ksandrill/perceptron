#include <iostream>
#include <chrono>
#include <fstream>
#include "math/Vector.h"
#include "math/Mat.h"
#include "Perceptron.h"

namespace {
    constexpr auto epoch_count = 3000;
    constexpr auto iterations = 4;

}

auto save_data_to_file(const Vector<float, epoch_count> &dataVector) -> void {
    auto outf = std::ofstream("error.txt");

    if (!outf) {
        std::cerr << "Uh oh, .txt could not be opened for writing!" << std::endl;
        exit(1);
    }
    for (auto error: dataVector) {
        outf << error << std::endl;
    }


}


auto main() -> int {
    auto begin = std::chrono::steady_clock::now();
    ////Perceptron<Type, input_neurones,output_neurones,first_neurones,second neurones>
    perceptronStuff::Perceptron<float, 2, 1, 3, 3> perceptron{};
    perceptron.initWeights(-1, 1);
    Mat<float, 4, 2> input{{0, 0},
                           {0, 1},
                           {1, 0},
                           {1, 1}};
    Vector<Vector<float, 1>, 4> output{{0},
                                       {1},
                                       {1},
                                       {1}};
    Vector<float, epoch_count> errorVector{};
    for (auto epoch = 0; epoch < epoch_count; ++epoch) {
        auto epoch_error = float{};
        for (auto i = 0; i < 4; ++i) {
            perceptron.setInputLayer(input[i]);
            perceptron.feedForward();
            perceptron.backProp(output[i]);
            epoch_error += perceptron.backProp(output[i]);
        }
        epoch_error /= iterations;
        errorVector[epoch] = epoch_error;

    }
    save_data_to_file(errorVector);
    std::cout << "epoch count: " << epoch_count << std::endl;
    for (auto i = 0; i < 4; ++i) {
        perceptron.setInputLayer(input[i]);
        std::cout << "input: " << perceptron.getInputLayer();
        perceptron.feedForward();
        std::cout << " output: " << perceptron.getOutputLayerOutput() << std::endl;


    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);\
    std::cout << "time: " << elapsed_time.count() << " ms\n";


    return 0;
}

