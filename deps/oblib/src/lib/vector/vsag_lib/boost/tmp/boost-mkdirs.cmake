# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/boost/source"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/boost/src/boost-build"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/boost"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/boost/tmp"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/boost/src/boost-stamp"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/boost/src"
  "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/boost/src/boost-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/boost/src/boost-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/vsag_lib/boost/src/boost-stamp${cfgdir}") # cfgdir has leading slash
endif()
