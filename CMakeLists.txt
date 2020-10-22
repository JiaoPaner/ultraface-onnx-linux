cmake_minimum_required(VERSION 3.15)
project(ultraface)

set(CMAKE_CXX_STANDARD 14)

string(APPEND CMAKE_C_FLAGS "-Wwrite-strings")

find_package(OpenCV REQUIRED)
include_directories(include/onnx include ${OpenCV_INCLUDE_DIRS})

link_directories(lib)

file(GLOB src src/*.cpp include/*.h src/*.c)
set(SRC ${src})

add_executable(ultraface ${SRC})
#add_library(ultraface SHARED ${SRC})

target_link_libraries(ultraface ${OpenCV_LIBS} onnxruntime)