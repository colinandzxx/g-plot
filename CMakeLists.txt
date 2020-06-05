cmake_minimum_required(VERSION 3.0.0)

SET(CMAKE_C_COMPILER /usr/bin/gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/g++)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set( ENV{CUDA_HOME} /usr/local/cuda )

set(CMAKE_SKIP_BUILD_RPATH TRUE) 

project(sphshabal VERSION 0.1.0 LANGUAGES CXX CUDA)


# cmake寻找cuda，这个要现在系统里面装好cuda，设置好cuda的环境参数啥的
FIND_PACKAGE(CUDA REQUIRED)

# set( ENV{CUDACXX} $ENV{CUDA_HOME}/bin/nvcc )
# set( ENV{PATH} $PATH:$CUDA_HOME/bin )

include(CTest)
enable_testing()

add_library(${PROJECT_NAME} SHARED shabal.cu)
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
