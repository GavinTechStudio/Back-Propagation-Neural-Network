cmake_minimum_required(VERSION 3.20)
project(BPNN)

set(CMAKE_CXX_STANDARD 17)

add_executable(BPNN main.cpp Utils.cpp Utils.h Config.h Net.cpp Net.h)
