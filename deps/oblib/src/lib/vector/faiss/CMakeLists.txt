cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

set(FAISS_LANGUAGES CXX)

project(faiss
  VERSION 1.9.0
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES ${FAISS_LANGUAGES})
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(OpenMP_gomp_LIBRARY "-fopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "avx2", "avx512", "sve".
option(FAISS_OPT_LEVEL "avx512")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." OFF)
option(FAISS_ENABLE_CUVS "Enable cuVS for GPU indexes." OFF)
option(FAISS_ENABLE_ROCM "Enable ROCm for GPU indexes." OFF)
option(FAISS_ENABLE_PYTHON "Build Python extension." OFF)
option(FAISS_ENABLE_C_API "Build C API." OFF)
option(FAISS_USE_LTO "Enable Link-Time optimization" OFF)
option(FAISS_TEST "faiss_test" OFF)

add_subdirectory(faiss)
set(OpenMP_omp_LIBRARY "-fopenmp")
set(OpenMP_gomp_LIBRARY "-fopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_C_FLAGS "-fopenmp")
set(OpenMP_C_LIB_NAMES "omp")

find_package(OpenMP REQUIRED)
add_library(faiss_lib ob_faiss_lib.cpp)
target_link_libraries(faiss_lib PRIVATE faiss_avx512)
target_link_libraries(faiss_lib PRIVATE OpenMP::OpenMP_CXX)
add_dependencies(faiss_lib faiss_avx512)


if (FAISS_TEST)
  add_executable(faiss_test_test_test faiss_test.cpp)
  target_link_libraries(faiss_test_test_test PRIVATE faiss_lib)
  # target_compile_options(faiss_test_test_test PRIVATE -O3)
  target_compile_options(faiss_lib PRIVATE -O2)
  target_compile_options(faiss_avx512 PRIVATE -O2)
  target_compile_options(faiss PRIVATE -O2)
  add_dependencies(faiss_test_test_test faiss_lib)
else()
  target_compile_options(faiss_lib PRIVATE -O3)
  target_compile_options(faiss_avx512 PRIVATE -O3)
  target_compile_options(faiss PRIVATE -O3)
endif()
