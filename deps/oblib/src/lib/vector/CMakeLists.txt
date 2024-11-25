
# add_subdirectory (llvm-project/openmp)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

project(faiss_vector_test_project)

add_subdirectory (vsag_lib)
add_subdirectory (faiss)

# add_executable(faiss_vector_test_test faiss_test.cpp)
# target_link_libraries(faiss_vector_test_test PRIVATE faiss_lib)
# add_dependencies(faiss_vector_test_test faiss_lib)