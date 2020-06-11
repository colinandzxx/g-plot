cmake_minimum_required(VERSION 3.10.0)
project(lib-sphshabal)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS ON)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CTEST_OUTPUT_ON_FAILURE ON)

# set(CMAKE_SKIP_BUILD_RPATH ON)

# libsphshabal
add_subdirectory(plot)

# -------------------
# test
# -------------------
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

enable_testing()

aux_source_directory(tests TEST_SRCS)

foreach(_test_file ${TEST_SRCS})
  get_filename_component(_test_name ${_test_file} NAME_WE)
  add_executable(${_test_name} ${_test_file})
  target_link_libraries(${_test_name} ${GTEST_BOTH_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(${_test_name} plot)
  target_include_directories(${_test_name} PUBLIC ${PROJECT_SOURCE_DIR}/plot/include)
  add_test(${_test_name} ${_test_name})
  # set_tests_properties(${_test_name} PROPERTIES TIMEOUT 5)
endforeach()
