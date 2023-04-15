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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB h_source_files "${CMAKE_SOURCE_DIR}/blas/*.h")
file(GLOB cpp_source_files "${CMAKE_SOURCE_DIR}/blas/*.cpp")
file(GLOB cu_source_files "${CMAKE_SOURCE_DIR}/blas/*.cu")

cuda_add_library(blas SHARED 
  ${h_source_files}
  ${cpp_source_files} 
  ${cu_source_files}
)

target_link_libraries(blas ${PYTHON_LIBRARIES} cudart)

set_target_properties(blas PROPERTIES PREFIX "")
