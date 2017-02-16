cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)

project(opencl_prototype)

file(GLOB_RECURSE opencl_prototype_SOURCES "src/*.cpp")
file(GLOB_RECURSE opencl_prototype_HEADERS "include/*.hpp")

include_directories(include)
add_executable(opencl_prototype ${opencl_prototype_SOURCES})
target_link_libraries(opencl_prototype OpenCL)
