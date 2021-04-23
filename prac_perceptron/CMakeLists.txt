cmake_minimum_required(VERSION 3.17)
project(prac_perceptron)

set(CMAKE_CXX_STANDARD 20)

include_directories(noBoostNumCpp)
add_executable(prac_perceptron main.cpp Perceptron/Perceptron.cpp Perceptron/Perceptron.h Perceptron/Layer.cpp Perceptron/Layer.h Perceptron/sigmoid.h)

