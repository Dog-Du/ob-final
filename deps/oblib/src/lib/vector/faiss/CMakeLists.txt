# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
# 调整最低版本需求# 调整最低版本需求
cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

set(FAISS_LANGUAGES CXX)

if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    # 替代 list(PREPEND ...) 的用法
    list(APPEND FAISS_LANGUAGES HIP)
    list(INSERT CMAKE_MODULE_PATH 0 "/opt/rocm/lib/cmake")
    list(INSERT CMAKE_PREFIX_PATH 0 "/opt/rocm")
  else()
    list(APPEND FAISS_LANGUAGES CUDA)
  endif()
endif()

if(FAISS_ENABLE_RAFT)
  include(cmake/thirdparty/fetch_rapids.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)

  # 替代 rapids_cuda_init_architectures 调用方式（如果需要）
  rapids_cuda_init_architectures(faiss)
  rapids_cuda_init_architectures(pyfaiss)
  rapids_cuda_init_architectures(faiss_c_library)
endif()

project(faiss
  VERSION 1.9.0
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES ${FAISS_LANGUAGES}
)

include(GNUInstallDirs)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)


# 添加模块路径
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# 配置选项
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." OFF)
option(FAISS_ENABLE_RAFT "Enable RAFT for GPU indexes." OFF)
option(FAISS_ENABLE_ROCM "Enable ROCm for GPU indexes." OFF)
option(FAISS_ENABLE_PYTHON "Build Python extension." OFF)
option(FAISS_ENABLE_C_API "Build C API." ON)
option(FAISS_USE_LTO "Enable Link-Time optimization" OFF)

# if(FAISS_ENABLE_GPU)
#   if(FAISS_ENABLE_ROCM)
#     enable_language(HIP)
#     add_definitions(-DUSE_AMD_ROCM)
#     find_package(HIP REQUIRED)
#     find_package(hipBLAS REQUIRED)
#     set(GPU_EXT_PREFIX "hip")
#   else ()
#     set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
#     enable_language(CUDA)
#     set(GPU_EXT_PREFIX "cu")
#   endif()
# endif()

# if(FAISS_ENABLE_RAFT AND NOT TARGET raft::raft)
#   find_package(raft COMPONENTS compiled distributed)
# endif()

include_directories (faiss)

# 添加子目录
add_subdirectory(faiss)

# if(FAISS_ENABLE_GPU)
#   if(FAISS_ENABLE_ROCM)
#     add_subdirectory(faiss/gpu-rocm)
#   else()
#     add_subdirectory(faiss/gpu)
#   endif()
# endif()

# if(FAISS_ENABLE_PYTHON)
#   add_subdirectory(faiss/python)
# endif()

if(FAISS_ENABLE_C_API)
  add_subdirectory(c_api)
endif()

add_subdirectory(demos)
add_subdirectory(benchs)
add_subdirectory(tutorial/cpp)

# install
install (DIRECTORY faiss
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install (DIRECTORY faiss_sve
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install (DIRECTORY faiss_avx2
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install (DIRECTORY faiss_avx512
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install (TARGETS faiss
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install (TARGETS faiss_avx2
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install (TARGETS faiss_avx512
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install (TARGETS faiss_sve
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

set(FAISS_BASE_DIR ${CMAKE_BINARY_DIR})
set(FAISS_LIB_DIR ${VSAG_BASE_DIR}/_deps)

add_library(ob_faiss_test SHARED ob_faiss_lib.cpp)
target_compile_options(ob_faiss_test PRIVATE -std=c++17)
target_include_directories(ob_faiss_test PRIVATE
                           ${FAISS_BASE_DIR}/faiss
                            )
target_compile_definitions(ob_faiss_test PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(ob_faiss_test PUBLIC faiss -static-libstdc++ -static-libgcc)
target_link_libraries(ob_faiss_test PUBLIC faiss_avx2 -static-libstdc++ -static-libgcc)
target_link_libraries(ob_faiss_test PUBLIC faiss_avx512 -static-libstdc++ -static-libgcc)
target_link_libraries(ob_faiss_test PUBLIC faiss_sve -static-libstdc++ -static-libgcc)
add_dependencies(ob_faiss_test faiss)
add_dependencies(ob_faiss_test faiss_avx2)
add_dependencies(ob_faiss_test faiss_avx512)
add_dependencies(ob_faiss_test faiss_sve)

# Create static library
add_library(ob_faiss_static_test STATIC ob_faiss_lib.cpp)
target_compile_options(ob_faiss_static_test PRIVATE -std=c++17)
target_compile_definitions(ob_faiss_static_test PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(ob_faiss_static_test PUBLIC
                           ${VSAG_BASE_DIR}/faiss)

add_dependencies(ob_faiss_static_test faiss)
add_dependencies(ob_faiss_static_test faiss_avx2)
add_dependencies(ob_faiss_static_test faiss_avx512)
add_dependencies(ob_faiss_static_test faiss_sve)

# 包含测试目标
# include(CTest)
# if(BUILD_TESTING)
#   add_subdirectory(tests)
#   add_subdirectory(perf_tests)
#   if(FAISS_ENABLE_GPU)
#     if(FAISS_ENABLE_ROCM)
#       add_subdirectory(faiss/gpu-rocm/test)
#     else()
#       add_subdirectory(faiss/gpu/test)
#     endif()
#   endif()
# endif()
