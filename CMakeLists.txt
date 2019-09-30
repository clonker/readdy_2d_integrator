cmake_minimum_required(VERSION 3.15)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

project(readdy_2d_integrator)

find_package(READDY REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_library(readdy_2d_integrator SHARED library.cpp library.h)
