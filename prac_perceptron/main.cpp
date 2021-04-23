#include <iostream>
#include "Perceptron/Perceptron.h"
#include "cmath"

#define PI 3.14159265
constexpr auto set_size = 50;
constexpr auto input_size = 1;
constexpr auto first_layer_size = 30;
constexpr auto second_layer_size = 20;
constexpr auto output_size = 1;
constexpr auto epoch_count = 10000;
constexpr auto lerning_rate = 0.04f;


void saveResults(const DataSet &data) {
    auto outf = std::ofstream("out.txt");
    auto inf = std::ofstream("in.txt");

    if (!outf) {
        std::cerr << "Uh oh, .txt could not be opened for writing!" << std::endl;
        exit(1);
    }
    for (const auto& elem: data) {
        inf << elem.first[0] << std::endl;
        outf << elem.second[0] << std::endl;
    }


}


int main() {
    auto begin = std::chrono::steady_clock::now();
    DataSet datum{};
    auto x = nc::linspace(-PI, PI, 50);
    for (int i = 0; i < set_size; ++i) {
        auto aux = Data();
        aux.first = nc::zeros<float>({1, input_size});
        aux.first = x[i];
        aux.second = nc::zeros<float>({1, output_size});
        aux.second = (std::sin(x[i]) + 1.f) / 2.f;
        datum.emplace_back(aux);
    }
    auto perceptron = new Perceptron(input_size, first_layer_size, second_layer_size, output_size);
    perceptron->initWeights(-1, 1);
    perceptron->train(datum, epoch_count, lerning_rate);
    auto res = perceptron->getTestResult(datum);
    saveResults(res);
    delete perceptron;

    auto end = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "time: " << elapsed_time.count() << " ms\n";
    return 0;
}
