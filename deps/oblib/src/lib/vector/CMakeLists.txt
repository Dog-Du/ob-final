
# add_subdirectory (llvm-project/openmp)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

add_subdirectory (vsag_lib)
set(tmp_cc ${CMAKE_C_COMPILER})
set(tmp_cxx ${CMAKE_CXX_COMPILER})
set(tmp_cc_flags ${CMAKE_C_FLAGS})
set(tmp_cxx_flags ${CMAKE_CXX_FLAGS})
set(tmp_cxx_link_flags ${CMAKE_CXX_LINK_FLAGS})
set(tmp_shared_link_flags ${CMAKE_SHARED_LINKER_FLAGS})
set(tmp_exe_link_flags ${CMAKE_EXE_LINKER_FLAGS})

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_FLAGS "-fopenmp -O3")
set(CMAKE_CXX_FLAGS "-fopenmp -O3")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,noexecstack")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,noexecstack")


add_subdirectory (faiss)


# add_executable(faiss_vector_test_test faiss_test.cpp)
# target_link_libraries(faiss_vector_test_test PRIVATE faiss_lib)
# add_dependencies(faiss_vector_test_test faiss_lib)

set(CMAKE_EXE_LINKER_FLAGS ${tmp_exe_link_flags})
set(CMAKE_SHARED_LINKER_FLAGS ${tmp_shared_link_flags})
set(CMAKE_CXX_LINK_FLAGS ${tmp_cxx_link_flags})
set(CMAKE_CXX_FLAGS ${tmp_cxx_flags})
set(CMAKE_C_FLAGS ${tmp_cc_flags})
set(CMAKE_C_COMPILER ${tmp_cc})
set(CMAKE_CXX_COMPILER ${tmp_cxx})
