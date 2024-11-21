add_subdirectory (vsag_lib)
add_subdirectory (faiss)

# 添加 faiss 目录到头文件搜索路径
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/faiss)

# cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

# # 添加子目录
# add_subdirectory(vsag_lib)
# add_subdirectory(faiss)

# # 添加 faiss 目录到头文件搜索路径
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/faiss)

# # 添加 vector 目录下的源文件
# add_executable(vector_example ob_vector_util.cpp)

# # 链接 faiss 库
# target_link_libraries(vector_example faiss_lib)

# # 安装头文件
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
#   DESTINATION include/vector
#   FILES_MATCHING PATTERN "*.h"
# )