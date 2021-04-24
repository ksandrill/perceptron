#include <iostream>
#include "NumCpp.hpp"
#include "Perceptron.h"
#include <dirent.h>

constexpr auto input_size = 2048;
constexpr auto first_layer_size = 700;
constexpr auto second_layer_size = 600;
constexpr auto output_layer_size = 2;
constexpr auto epoch_count = 10000;
constexpr auto learning_rate = 0.0003;
#define PI 3.14159265

void saveError(const NdArrayF &data) {
    auto outf = std::ofstream("error.txt");

    if (!outf) {
        std::cerr << "Uh oh, .txt could not be opened for writing!" << std::endl;
        exit(1);
    }
    for (const auto &elem: data) {
        outf << elem << std::endl;
    }

}


void saveResults(const DataSet &data) {
    auto sinOut = std::ofstream("SinOut.txt");
    auto CosOut = std::ofstream("CosOut.txt");

    auto inf = std::ofstream("in.txt");

    if (!sinOut) {
        std::cerr << "Uh oh, .txt could not be opened for writing!" << std::endl;
        exit(1);
    }
    for (const auto &elem: data) {
        inf << elem.first[0] << std::endl;
        sinOut << elem.second[0] << std::endl;
        CosOut << elem.second[1] << std::endl;

    }

}


auto listDir(const char *path) -> std::vector<std::string> {
    std::vector<std::string> dirList{};
    struct dirent *entry;
    DIR *dir = opendir(path);
    while ((entry = readdir(dir)) != nullptr) {
        auto item = entry->d_name;
        if (!item || item[0] == '.') {
            continue;
        }
        dirList.emplace_back(item);
    }
    closedir(dir);
    return dirList;
}


void addClassToDataSet(DataSet &dataSet, const std::string &classDirectory, int classIndex) {
    auto classMemberList = listDir(classDirectory.c_str());
    for (const auto &memberName: classMemberList) {
        auto fileName = std::string();
        fileName.append(classDirectory);
        fileName.append("/");
        fileName.append(memberName);
        auto file = std::ifstream(fileName);
        auto inputIter = 0;
        auto aux = Data();
        aux.first = nc::zeros<float>({1, input_size});
        aux.second = nc::zeros<float>({1, output_layer_size});
        ///std::cout << fileName << std::endl;
        for (std::string line; getline(file, line) && inputIter < input_size;) {
            aux.first[inputIter] = std::stof(line);
            ++inputIter;
        }
        for (auto i = 0; i < output_layer_size; ++i) {
            aux.second[i] = i == classIndex ? 1.f : 0.f;
        }
        dataSet.emplace_back(aux);
    }


}

DataSet getDataSet(const std::string &pathToSet) {
    DataSet datum{};
    auto classNameList = listDir(pathToSet.c_str());
    auto i = 0;
    for (const auto &className: classNameList) {
        auto classPath = std::string();
        classPath.append(pathToSet);
        classPath.append("/");
        classPath.append(className);
        ///std::cout << classPath << std::endl;
        addClassToDataSet(datum, classPath, i);
        ++i;
    }
    return datum;


}


int main() {
    auto x = nc::linspace(-PI/2, PI/2, 50);
//    DataSet datum{};
//    for (int i = 0; i < 50; ++i) {
//        auto aux = Data();
//        aux.first = nc::zeros<float>({1, input_size});
//        aux.first = x[i];
//        aux.second = nc::zeros<float>({1, output_layer_size});
//        aux.second[0] = (std::sin(x[i]) + 1.f) / 2.f;
//        aux.second[1] = ((std::atan(x[i] + PI/2) / PI));
//        datum.emplace_back(aux);
//    }
    auto datum = getDataSet("../data/train");
    auto percteptron = new Perceptron(input_size, first_layer_size, second_layer_size, output_layer_size);
    percteptron->initWeights(-1, 1);
    auto mseVector = percteptron->train(datum, epoch_count, learning_rate);
    auto res = percteptron->getTestResult(datum);
    saveError(mseVector);
    ///saveResults(res);
    delete percteptron;


    return 0;
}
