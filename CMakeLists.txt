cmake_minimum_required(VERSION 2.8.12)
project(BPNN)

set(CMAKE_CXX_STANDARD 11)

add_executable(BPNN main.cpp lib/Utils.cpp lib/Utils.h lib/Config.h lib/Net.cpp lib/Net.h)
