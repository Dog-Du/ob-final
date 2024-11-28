# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.23.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(nlohmann_json-populate NONE)



include(ExternalProject)
ExternalProject_Add(nlohmann_json-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz" "http://vsagcache.oss-rg-china-mainland.aliyuncs.com/nlohmann_json/v3.11.3.tar.gz" "DOWNLOAD_NO_PROGRESS" "1" "INACTIVITY_TIMEOUT" "5" "TIMEOUT" "30"
                    SOURCE_DIR          "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/faiss/_deps/nlohmann_json-src"
                    BINARY_DIR          "/home/user/oceanbase-2024/oceanbase/deps/oblib/src/lib/vector/faiss/_deps/nlohmann_json-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


