# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/spdlog/source"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/spdlog/src/spdlog-build"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/spdlog"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/spdlog/tmp"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/spdlog/src/spdlog-stamp"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/spdlog/src"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/spdlog/src/spdlog-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/spdlog/src/spdlog-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/spdlog/src/spdlog-stamp${cfgdir}") # cfgdir has leading slash
endif()
