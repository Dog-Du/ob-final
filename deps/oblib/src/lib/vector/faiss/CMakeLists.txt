# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(faiss
  VERSION 1.9.0
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# set(CMAKE_C_COMPILER "gcc")
# set(CMAKE_CXX_COMPILER "g++")

# Valid values are "generic", "avx2", "avx512", "sve".
option(FAISS_OPT_LEVEL "avx512") # 直接只开avx512
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." OFF)
option(FAISS_ENABLE_RAFT "Enable RAFT for GPU indexes." OFF)
option(FAISS_ENABLE_ROCM "Enable ROCm for GPU indexes." OFF)
option(FAISS_ENABLE_PYTHON "Build Python extension." OFF)
option(FAISS_ENABLE_C_API "Build C API." OFF)
option(FAISS_USE_LTO "Enable Link-Time optimization" OFF)

set(FAISS_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}) # /deps/oblib/src/lib/vector/faiss)
set(FAISS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message("FAISS_LIB_DIR : " ${FAISS_LIB_DIR})
message("FAISS_LIB_DIR : " ${FAISS_LIB_DIR})
message("FAISS_LIB_DIR : " ${FAISS_LIB_DIR})

message("FAISS_SOURCE_DIR  : " ${FAISS_SOURCE_DIR})
message("FAISS_SOURCE_DIR  : " ${FAISS_SOURCE_DIR})
message("FAISS_SOURCE_DIR  : " ${FAISS_SOURCE_DIR})
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
message(STATUS "Parent Source Dir: ${PARENT_DIR}")
message(STATUS "Parent Source Dir: ${PARENT_DIR}")
message(STATUS "Parent Source Dir: ${PARENT_DIR}")

# add_compile_options (-fopenmp)
# add_compile_options (-lomp)
# add_compile_options (-lgomp)

# add_compile_options (-pthread)
# add_compile_options (-lpthread)
# add_compile_options (-lgpthread)

add_subdirectory(faiss)
# add_subdirectory(c_api)
# add_subdirectory(demos)
# add_subdirectory(benchs)
# add_subdirectory(tutorial/cpp)

message(${FAISS_LIB_DIR}/faiss/libfaiss.a)
message(${FAISS_LIB_DIR}/faiss/libfaiss_avx512.a)

# link_libraries(${FAISS_LIB_DIR}/faiss/libfaiss.a)
# link_libraries(${FAISS_LIB_DIR}/faiss/libfaiss_avx512.a)

include_directories(${FAISS_SOURCE_DIR})
include_directories(${FAISS_SOURCE_DIR}/faiss)
include_directories(${PARENT_DIR})

link_directories(${FAISS_LIB_DIR})
link_directories(${FAISS_LIB_DIR}/faiss)
link_directories(${PARENT_DIR})

# 添加 faiss_lib 库
add_library(faiss_lib STATIC ob_faiss_lib.cpp)
target_compile_options(faiss_lib PRIVATE -O3)
# target_compile_options(faiss_lib PRIVATE -fPIC)

# 添加头文件目录
target_include_directories(faiss_lib PRIVATE ${FAISS_SOURCE_DIR} ${FAISS_SOURCE_DIR}/faiss ${PARENT_DIR})
target_include_directories(faiss_lib PRIVATE
   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>/faiss $<BUILD_INTERFACE:${PARENT_DIR}>)

target_link_directories(
   faiss_lib PRIVATE ${FAISS_LIB_DIR}/faiss ${FAISS_LIB_DIR} ${PARENT_DIR}
)


# target_link_options(faiss_lib PRIVATE -Wl,--exclude-libs,ALL)

# set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} -fopenmp")
# set(OpenMP_gomp_LIBRARY "-Wl,-Bstatic -lgomp -Wl,-Bdynamic")

target_link_libraries(faiss_lib PRIVATE libfaiss_avx512.a)
# target_link_libraries(faiss_lib PUBLIC libfaiss_c.a)
# target_link_libraries(faiss_lib PRIVATE faiss)
# target_link_libraries(faiss_lib PUBLIC libfaiss_avx2.a)
target_link_libraries(faiss_lib PRIVATE faiss_avx512)
target_link_libraries(faiss_lib PRIVATE ${OpenMP_gomp_LIBRARY})

# target_link_libraries(faiss_lib PUBLIC ${FAISS_LIB_DIR}/c_api/libfaiss_c.a)
# target_link_libraries(faiss_lib PRIVATE ${FAISS_LIB_DIR}/faiss/libfaiss.a)
# target_link_libraries(faiss_lib PUBLIC ${FAISS_LIB_DIR}/faiss/libfaiss_avx2.a)
target_link_libraries(faiss_lib PRIVATE ${FAISS_LIB_DIR}/faiss/libfaiss_avx512.a)

# target_link_libraries(faiss_lib PUBLIC faiss_c)
# target_link_libraries(faiss_lib PRIVATE faiss)
# target_link_libraries(faiss_lib PUBLIC faiss_avx2)
# target_link_libraries(faiss_lib PRIVATE faiss_avx512)

# add_dependencies(faiss_lib faiss_c)
# add_dependencies(faiss_lib faiss)
# add_dependencies(faiss_lib faiss_avx2)
add_dependencies(faiss_lib faiss_avx512)
# target_link_libraries(faiss_lib faiss_sve) # 可能是因为不支持sve，但是不清楚为什么到现在才报错。

# 安装 faiss 库和头文件
install(DIRECTORY ${FAISS_SOURCE_DIR}/
  DESTINATION include/vector/faiss
  FILES_MATCHING PATTERN "*.h"
)

# install (TARGETS faiss_lib
# 	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
# add_executable(faiss_test_test faiss_test.cpp)
# target_link_libraries(faiss_test_test PRIVATE faiss_lib)
# add_dependencies(faiss_test_test faiss_lib)
