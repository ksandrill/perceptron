#include <iostream>
#include "NumCpp.hpp"
#include "Perceptron.h"
#include <dirent.h>
#include <limits>

constexpr auto input_size = 15;
constexpr auto first_layer_size = 70;
constexpr auto second_layer_size = 63;
constexpr auto output_layer_size = 2;
constexpr auto epoch_count = 5000;
constexpr auto learning_rate = 0.044;
constexpr auto low_weight = -1.9;
constexpr auto high_weight = 1.9;


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

void saveError(const NdArrayF &data, const std::string &fileName) {
    auto outf = std::ofstream("../pyWrapper/" + fileName);

    if (!outf) {
        std::cerr << "Uh oh, .txt could not be opened for writing!" << std::endl;
        exit(1);
    }
    for (const auto &elem: data) {
        outf << elem << std::endl;
    }

}


void saveTestErrors(const std::pair<std::vector<NdArrayF>, std::vector<float>> &data) {
    auto sinOut = std::ofstream("../pyWrapper/G_dev.txt");
    auto CosOut = std::ofstream("../pyWrapper/kgf.txt");
    auto testMse = std::ofstream("../pyWrapper/testMse.txt");

    auto inf = std::ofstream("in.txt");

    if (!sinOut) {
        std::cerr << "Uh oh, .txt could not be opened for writing!" << std::endl;
        exit(1);
    }
    for (const auto &elem: data.first) {
        sinOut << elem[0] << std::endl;
        CosOut << elem[1] << std::endl;

    }
    for (const auto &elem:data.second) {
        testMse << elem << std::endl;
    }

}


auto parse_csv(
        std::istream &file,
        char delimiter,
        unsigned target_columns_count) -> DataSet {
    auto line = std::string();
    auto result = DataSet();

    while (std::getline(file, line)) {
        auto line_buffer = std::istringstream(line);
        auto token = std::string();
        auto row = std::vector<float>();

        while (std::getline(line_buffer, token, delimiter)) {
            row.push_back(std::stod(token));
        }

        auto array_row = nc::NdArray(row);
        result.emplace_back(
                array_row[nc::Slice(-target_columns_count)],
                array_row[nc::Slice(-target_columns_count, array_row.size())]);
    }

    return result;
}

void fixNanDataSet(DataSet &datum) {
    for (auto &i : datum) {
        for (float &j : i.first) {
            j = std::isnan(j) ? 0.f : j;
        }

    }
}


void normalInput(DataSet &datum) {
    NdArrayF min = nc::zeros<float>({1, input_size});
    NdArrayF max = nc::zeros<float>({1, input_size});
    for (auto i = 0; i < input_size; ++i) {
        max[i] = std::numeric_limits<float>::lowest();
        min[i] = std::numeric_limits<float>::max();
    }
    for (auto &i : datum) {
        for (auto j = 0; j < i.first.size(); ++j) {
            if (!std::isnan(i.first[j])) {
                max[j] = max[j] < i.first[j] ? i.first[j] : max[j];
                min[j] = min[j] > i.first[j] ? i.first[j] : min[j];

            }

        }
    }
    for (auto &i : datum) {
        for (auto j = 0; j < i.first.size(); ++j) {
            if (!std::isnan(i.first[j])) {
                i.first[j] = (i.first[j] - min[j]) / (max[j] - min[j]);
            }

        }

    }


}


void normalOutput(DataSet &datum) {
    NdArrayF min = nc::zeros<float>({1, output_layer_size});
    NdArrayF max = nc::zeros<float>({1, output_layer_size});
    for (auto i = 0; i < output_layer_size; ++i) {
        max[i] = std::numeric_limits<float>::lowest();
        min[i] = std::numeric_limits<float>::max();
    }
    for (auto &i : datum) {
        for (auto j = 0; j < i.second.size(); ++j) {
            if (!std::isnan(i.second[j])) {
                max[j] = max[j] < i.second[j] ? i.second[j] : max[j];
                min[j] = min[j] > i.second[j] ? i.second[j] : min[j];

            }

        }
    }
    for (auto &i : datum) {
        for (auto j = 0; j < i.second.size(); ++j) {
            if (!std::isnan(i.second[j])) {
                i.second[j] = (i.second[j] - min[j]) / (max[j] - min[j]);
            }

        }

    }


}


int main() {
    auto ifs = std::ifstream("../trainSet.csv");
    auto trainDatum = parse_csv(ifs, ',', 2);
    fixNanDataSet(trainDatum);
    normalInput(trainDatum);
    normalOutput(trainDatum);
    ////auto trainDatum = getDataSet("../data/train");
    auto percteptron = std::make_unique<Perceptron>(input_size, first_layer_size, second_layer_size, output_layer_size);
    percteptron->initWeights(low_weight, high_weight);
    auto trainMseVector = percteptron->train(trainDatum, epoch_count, learning_rate);
    ifs = std::ifstream("../testSet.csv");
    auto testDatum = parse_csv(ifs, ',', 2);
    fixNanDataSet(testDatum);
    normalInput(testDatum);
    normalOutput(testDatum);
    auto resDev = percteptron->getTestResult(testDatum);
    saveError(trainMseVector, "trainMse.txt");
    saveTestErrors(resDev);
    auto rmseG = 0.f;
    auto rmseKGF = 0.f;
    auto errorList = resDev.first;
    for(auto i =0; i < errorList.size();++i){
        rmseG += errorList[i][0]*errorList[i][0];
        rmseKGF+= errorList[i][1]*errorList[i][1];
    }
    rmseKGF = nc::sqrt(rmseKGF/errorList.size());
    rmseG = nc::sqrt(rmseG/errorList.size());
    std::cout << "rmse KGF:" << rmseKGF << std::endl;
    std::cout << "rmse G_total" << rmseG << std::endl;



    return 0;
}
