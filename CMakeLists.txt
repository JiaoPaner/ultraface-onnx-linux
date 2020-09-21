cmake_minimum_required(VERSION 3.15)
project(retinaface)

set(CMAKE_CXX_STANDARD 14)

#string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
#string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS "-Wwrite-strings")

find_package(OpenCV REQUIRED)
set(ONNX_DIR )
include_directories(include/onnx include ${OpenCV_INCLUDE_DIRS})
link_directories(lib)

file(GLOB src src/*.cpp include/*.h src/*.c)
set(SRC ${src})

add_executable(retinaface ${SRC})
#add_library(cv_algorithm SHARED ${SRC})

target_link_libraries(retinaface ${OpenCV_LIBS} onnxruntime)