cmake_minimum_required(VERSION 2.6)

project(yolov5n_v6_prune)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)


find_package(CUDA)
find_package(OpenCV 4.1 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include ${OpenCV_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/opt/TensorRT/include)
link_directories(/opt/TensorRT/lib)


#cuda_add_library(nanodetplugin SHARED ${PROJECT_SOURCE_DIR}/decode.cu)
#target_link_libraries(nanodetplugin nvinfer cudart)

#cuda_add_executable(yolov5n_v6_prune ${PROJECT_SOURCE_DIR}/yolov5n_v6_prune.cpp ${PROJECT_SOURCE_DIR}/yolov5n_v6_prune_plugin.cu)
cuda_add_executable(yolov5n_v6_prune ${PROJECT_SOURCE_DIR}/yolov5n_v6_prune_paddle.cpp ${PROJECT_SOURCE_DIR}/yolov5n_v6_prune_plugin.cu)
target_link_libraries(yolov5n_v6_prune nvinfer)
target_link_libraries(yolov5n_v6_prune cudart ${OpenCV_LIBS})
#target_link_libraries(naodet nanodetplugin)

add_definitions(-O2 -pthread)

