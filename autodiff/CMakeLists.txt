cmake_minimum_required(VERSION 3.6)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.9 REQUIRED)
find_package(PythonLibs 3.9 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
    )

link_directories(
    /usr/local/cuda/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

cuda_add_library(tensorlib SHARED 
  "tensorlib/tensor.h"
  "tensorlib/tensor.cpp"
  "tensorlib/kernel.cu"
  "tensorlib/tensor.cu"
  "tensorlib/bindings.cpp"
)

target_link_libraries(tensorlib ${PYTHON_LIBRARIES} cudart)

set_target_properties(tensorlib PROPERTIES PREFIX "")
