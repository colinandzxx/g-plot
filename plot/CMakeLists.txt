cmake_minimum_required(VERSION 3.10.0)
project(
  plot
  VERSION 0.1.0
  LANGUAGES C)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# set(CUDA_CUDART_LIBRARY /usr/local/cuda/lib64/libcudart.so) set(ENV{CUDA_HOME}
# /usr/local/cuda)

# enable CUDA
enable_language(CUDA)
find_package(CUDA)

# set FLAGS C++/CUDA
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_61 --gpu-code=compute_61 --compiler-options \"-Wall -Wfatal-errors -Ofast -DOPENCV -DGPU -DCUDNN -fPIC\""
)

# source directory aux_source_directory(src DIR_SRCS)
file(GLOB DIR_CU_SRCS src/*.cu)
file(GLOB DIR_SRCS src/*.c)

# add library file add_library(sphshabal SHARED ${DIR_SRCS})
add_library(plot_cu OBJECT ${DIR_CU_SRCS})
add_library(plot SHARED ${DIR_SRCS} $<TARGET_OBJECTS:plot_cu>)

# add_library(${PROJECT_NAME}_shared SHARED shabal.cu)
# add_library(${PROJECT_NAME}_static STATIC shabal.cu)

# set_target_properties(sphshabal_shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
# set_target_properties(sphshabal_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
# set_target_properties(sphshabal_static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
# set_target_properties(sphshabal_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

# head file path
target_include_directories(plot_cu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(plot PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
