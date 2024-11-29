#include <hdf5.h>
#include <chrono>
#include <iostream>
#include <vector>
#include "ob_faiss_lib.h"

using namespace std;

// 读取HDF5文件中的数据
void load_data_from_hdf5(
        const char* filename,
        vector<float>& database,
        vector<float>& queries,
        vector<vector<int>>& true_neighbors) {
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    // 读取数据库向量
    hid_t dataset_db = H5Dopen(file_id, "database", H5P_DEFAULT);
    hid_t space_db = H5Dget_space(dataset_db);
    hsize_t db_size[2];
    H5Sget_simple_extent_dims(space_db, db_size, NULL);
    int num_db = db_size[0];
    int dim_db = db_size[1];

    database.resize(num_db * dim_db);
    H5Dread(dataset_db,
            H5T_NATIVE_FLOAT,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            database.data());

    // 读取查询向量
    hid_t dataset_query = H5Dopen(file_id, "queries", H5P_DEFAULT);
    hid_t space_query = H5Dget_space(dataset_query);
    hsize_t query_size[2];
    H5Sget_simple_extent_dims(space_query, query_size, NULL);
    int num_query = query_size[0];
    int dim_query = query_size[1];

    queries.resize(num_query * dim_query);
    H5Dread(dataset_query,
            H5T_NATIVE_FLOAT,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            queries.data());

    // 读取真实邻居
    hid_t dataset_true_neighbors =
            H5Dopen(file_id, "true_neighbors", H5P_DEFAULT);
    hid_t space_true_neighbors = H5Dget_space(dataset_true_neighbors);
    hsize_t true_neighbors_size[2];
    H5Sget_simple_extent_dims(space_true_neighbors, true_neighbors_size, NULL);
    int num_true_neighbors = true_neighbors_size[0];
    int k_neighbors = true_neighbors_size[1];

    true_neighbors.resize(num_true_neighbors);
    vector<int> temp_neighbors(k_neighbors);
    for (int i = 0; i < num_true_neighbors; ++i) {
        H5Dread(dataset_true_neighbors,
                H5T_NATIVE_INT,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                temp_neighbors.data());
        true_neighbors[i] = temp_neighbors;
    }

    // 关闭文件
    H5Dclose(dataset_db);
    H5Dclose(dataset_query);
    H5Dclose(dataset_true_neighbors);
    H5Fclose(file_id);
}

// 计算召回率
float calculate_recall_at_k(
        const vector<int>& I,
        const vector<vector<int>>& true_neighbors,
        int k) {
    int correct_count = 0;
    for (int i = 0; i < k; ++i) {
        if (find(true_neighbors.begin(), true_neighbors.end(), I[i]) !=
            true_neighbors.end()) {
            correct_count++;
        }
    }
    return static_cast<float>(correct_count) / k;
}

// 主函数
int main() {
    const char* hdf5_file =
            "/home/user/oceanbase-2024/oceanbase/ann-benchmarks/data/sift-128-euclidean.hdf5";
    int k = 10; // 计算 Recall@10

    // 载入数据
    vector<float> database, queries;
    vector<vector<int>> true_neighbors;
    load_data_from_hdf5(hdf5_file, database, queries, true_neighbors);

    // 生成FAISS索引
    int dimension = 128;
    obvectorlib::VectorIndexPtr index_handler;
    obvectorlib::create_index(
            index_handler,
            obvectorlib::HNSW_TYPE,
            "float32",
            "l2",
            dimension,
            32,
            400,
            64);


    obvectorlib::add_index(index_handler, database.data(), int64_t *ids, dimension, database.size())
    // 查询时间测量
    auto start = chrono::high_resolution_clock::now();
    vector<int> I(k);
    vector<float> D(k);
    for (int i = 0; i < queries.size() / dimension; ++i) {
        index.search(&queries[i * dimension], 1, k, D.data(), I.data());
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float> duration = end - start;

    // 计算召回率
    float recall_at_k = 0.0;
    for (int i = 0; i < queries.size() / dimension; ++i) {
        recall_at_k += calculate_recall_at_k(I, true_neighbors, k);
    }
    recall_at_k /= queries.size() / dimension;

    // 输出结果
    cout << "Recall@10: " << recall_at_k << endl;
    cout << "Total query time: " << duration.count() << " seconds" << endl;
    cout << "Average query time per query: "
         << duration.count() / (queries.size() / dimension) << " seconds"
         << endl;

    return 0;
}
