cmake_minimum_required(VERSION 3.10)

project(yolo)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
# add_definitions("-Wall -g")
# find_package(CUDA REQUIRED)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/local/TensorRT-8.6.1.6/include)
link_directories(/usr/local/TensorRT-8.6.1.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/comm)

add_executable(yolo ${PROJECT_SOURCE_DIR}/test/test.cpp ${PROJECT_SOURCE_DIR}/src/yolo.cpp)

target_link_libraries(yolo nvinfer)
target_link_libraries(yolo nvinfer_plugin)
target_link_libraries(yolo cudart)

target_link_libraries(yolo ${OpenCV_LIBS})

add_definitions(-O2 -pthread)