#include <omp.h>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <sstream>
#include "ob_faiss_lib.h"
int main() {
    obvectorlib::VectorIndexPtr index_handler;
    assert(obvectorlib::create_index(
                   index_handler,
                   obvectorlib::IndexType::HNSW_TYPE,
                   "float32",
                   "l2",
                   2,
                   10,
                   300,
                   10) == 0);

    int size = 1;
    int64_t ids[] = {10001};
    float vector_list[][2] = {{1, 1}};

    assert(obvectorlib::add_index(
                   index_handler, (float*)vector_list, ids, 2, size) == 0);

    std::fstream file("index.data", std::ios_base::out);
    assert(obvectorlib::fserialize(index_handler, file) == 0);
    file.close();
    std::fstream file1("index.data", std::ios_base::in);
    assert(obvectorlib::fserialize(index_handler, file1) == 0);
    file1.close();
    return 0;
}