cmake_minimum_required(VERSION 3.10)

project(yolo)

# 使用动态库链接到CUDA
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_BUILD_TYPE Debug)
# add_definitions("-Wall -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(TENSORRT_DIR /usr/local/TensorRT-8.6.1.6)
find_package(OpenCV)


include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TENSORRT_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/comm
    ${OpenCV_INCLUDE_DIRS}
)
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${TENSORRT_DIR}/lib
    ${OpenCV_LIB_DIR}    
)

add_executable(yolo ${PROJECT_SOURCE_DIR}/test/test.cpp ${PROJECT_SOURCE_DIR}/src/yolo.cpp)

target_link_libraries(yolo 
    nvinfer
    nvinfer_plugin
    cudart
    ${OpenCV_LIBS}
    -lstdc++fs
)

add_definitions(-O2 -pthread)