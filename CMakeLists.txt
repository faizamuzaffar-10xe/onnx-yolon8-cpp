cmake_minimum_required(VERSION 3.10)
project(cpp_onnx)

set(CMAKE_CXX_STANDARD 14)

# Set ONNX Runtime path
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.17.0)

# Include and link ONNX Runtime
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# add_executable(cpp_onnx main.cpp)
# target_link_libraries(cpp_onnx onnxruntime)



set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# add_executable(opencvTest main.cpp)
# target_link_libraries(opencvTest ${OpenCV_LIBS})

add_executable(cpp_onnx main.cpp)
target_link_libraries(cpp_onnx onnxruntime ${OpenCV_LIBS})