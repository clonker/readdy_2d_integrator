cmake_minimum_required(VERSION 3.15)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

project(readdyextension)
set(CMAKE_CXX_STANDARD 17)

find_package(READDY REQUIRED)
add_subdirectory(lib/pybind11)

pybind11_add_module(${PROJECT_NAME} binding.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC "${READDY_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} PRIVATE ${READDY_LIBRARIES})
