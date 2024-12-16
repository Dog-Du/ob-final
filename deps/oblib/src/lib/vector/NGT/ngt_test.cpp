#include <omp.h>
#include <stdint.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <random>
#include <sstream>
#include <vector>

#include "ob_ngt_lib.h"

std::minstd_rand r(789);

void
generate_vector(std::vector<float>& vector, int64_t dim) {
    vector.clear();
    vector.reserve(dim);

    for (int64_t i = 0; i < dim; ++i) {
        vector.push_back(r() % 100);
    }
}

void
generate_vector_list(std::vector<float>& vector_list,
                     std::vector<int64_t>& ids,
                     int64_t dim,
                     int64_t n) {
    vector_list.clear();
    vector_list.reserve(n * dim);

    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            vector_list.push_back(r() % 100);
        }
        ids.push_back(10000 + i);
    }
}

void
generate_data_row(char* data, uint32_t length) {
    for (uint32_t i = 0; i < length; ++i) {
        data[i] = r() % ('z' - 'a') + 'a';
    }
}

void
generate_datas(char*& data, uint32_t length, int n) {
    data = (char*)malloc(length * n);

    int offset = 0;
    for (int i = 0; i < n; ++i) {
        generate_data_row(data + offset, length);
        offset += length;
    }
}

void
free_datas(char**& data, uint32_t*& length, int n) {
    for (int i = 0; i < n; ++i) {
        free(data[i]);
    }
    free(length);
    free(data);
    data = nullptr;
    length = nullptr;
}

void
write_index(obvectorlib::VectorIndexPtr& index_handler) {
    std::ofstream file("index.data");
    assert(obvectorlib::fserialize(index_handler, file) == 0);
}

void
read_index(obvectorlib::VectorIndexPtr& index_handler) {
    assert(obvectorlib::delete_index(index_handler) == 0);
    assert(
        obvectorlib::create_index(
            index_handler, obvectorlib::IndexType::HNSW_TYPE, "float32", "l2", 128, 10, 300, 10) ==
        0);

    std::ifstream file("index.data");
    assert(obvectorlib::fdeserialize(index_handler, file) == 0);
}

void
restart_index(obvectorlib::VectorIndexPtr& index_handler) {
    write_index(index_handler);
    read_index(index_handler);
}

int
main() {
    obvectorlib::VectorIndexPtr index_handler;
    int64_t dim = 128;
    int64_t size = 1000000;
    int64_t index_size;
    int64_t topk = 10;
    char* data = nullptr;
    uint32_t length = 5;
    system("rm -rf ./ngt_index");

    std::vector<float> vector_list;
    std::vector<int64_t> ids;
    assert(
        obvectorlib::create_index(
            index_handler, obvectorlib::IndexType::HNSW_TYPE, "float32", "l2", dim, 10, 300, 10) ==
        0);

    generate_vector_list(vector_list, ids, dim, size);
    generate_datas(data, length, size);

    restart_index(index_handler);
    std::cout << "restart index sucessfully" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < ids.size(); ++i) {
        assert(obvectorlib::add_index(index_handler,
                                      vector_list.data() + dim * i,
                                      ids.data() + i,
                                      dim,
                                      1,
                                      data,
                                      length) == 0);

        if (i % 100000 == 0) {
            std::ofstream file("index.data");
            assert(obvectorlib::fserialize(index_handler, file) == 0);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "add_index cost time : " << duration.count() << "ms" << std::endl;

    std::cout << "add_index sucessfully" << std::endl;

    vector_list.clear();
    ids.clear();
    free(data);

    restart_index(index_handler);

    assert(obvectorlib::get_index_number(index_handler, index_size) == 0);
    std::cout << "index_number : " << index_size << std::endl;

    std::cout << "restart index successfully" << std::endl;

    auto query_index = [&](int topk) {
        for (int64_t i = 0; i < 100; ++i) {
            generate_vector(vector_list, dim);
            const float* dist = nullptr;
            const int64_t* ids = nullptr;
            int64_t result_size = 0;
            char** data = nullptr;
            uint32_t data_length = 0;

            assert(obvectorlib::knn_search(index_handler,
                                           vector_list.data(),
                                           dim,
                                           topk,
                                           dist,
                                           ids,
                                           result_size,
                                           0,
                                           data,
                                           data_length) == 0);

            std::cout << "query result size : " << result_size << " :: ";
            std::cout << std::endl;

            free(data);
            free((void*)ids);
            free((void*)dist);
        }
        std::cout << "query sucessfully" << std::endl;
    };

    query_index(topk);

    restart_index(index_handler);

    assert(obvectorlib::get_index_number(index_handler, index_size) == 0);
    std::cout << "index_number : " << index_size << std::endl;
    std::cout << "restart index successfully" << std::endl;

    topk = 10000;
    query_index(topk);

    std::cout << "query sucessfully" << std::endl;

    std::minstd_rand rng(666);

    for (int i = 0; i < 100000; ++i) {
        switch (rng() % 6) {
            case 0: {
                restart_index(index_handler);
                std::cout << "restart_index successfully" << std::endl;
            } break;
            case 1: {
                query_index(10);
                std::cout << "query topk == 10 successfully" << std::endl;
            } break;
            case 2: {
                query_index(10000);
                std::cout << "query topk == 10'000 successfully" << std::endl;
            } break;
            case 3: {
                int64_t x;
                assert(obvectorlib::get_index_number(index_handler, x) == 0);
                std::cout << "get_index_number : " << x << "  successfully" << std::endl;
            } break;
            case 4: {
                write_index(index_handler);
                std::cout << "write_index successfully" << std::endl;
            } break;
            case 5: {
                read_index(index_handler);
                std::cout << "read_index successfully" << std::endl;
            } break;
        }
    }
    return 0;
}