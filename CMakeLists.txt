cmake_minimum_required(VERSION 3.10.0)
project(sphshabal VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_CUDART_LIBRARY /usr/local/cuda/lib64/libcudart.so)
set(ENV{CUDA_HOME} /usr/local/cuda)

set(CMAKE_SKIP_BUILD_RPATH ON) 

# enable CUDA
enable_language(CUDA)
find_package(CUDA)

# set FLAGS
# C++和CUDA的编译参数，可选。
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 --gpu-architecture=compute_61 --gpu-code=compute_61 --compiler-options \"-Wall -Wfatal-errors -Ofast -DOPENCV -DGPU -DCUDNN -fPIC\"")

include(CTest)
enable_testing()

add_library(${PROJECT_NAME} SHARED sph_shabal.cu)
# target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# add_library(${PROJECT_NAME}_shared SHARED shabal.cu)
# add_library(${PROJECT_NAME}_static STATIC shabal.cu)

# set_target_properties(sphshabal_shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
# set_target_properties(sphshabal_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
# set_target_properties(sphshabal_static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
# set_target_properties(sphshabal_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(mytest test.cpp)
target_link_libraries(mytest ${PROJECT_NAME})
