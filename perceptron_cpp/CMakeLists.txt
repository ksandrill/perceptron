cmake_minimum_required(VERSION 3.17)
project(perceptron_cpp)

set(CMAKE_CXX_STANDARD 20)

add_executable(perceptron_cpp main.cpp Perceptron.h math/Vector.h math/Mat.h math/Util.h)