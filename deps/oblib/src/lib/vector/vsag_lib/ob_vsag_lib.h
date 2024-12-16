#ifndef OB_VSAG_LIB_H
#define OB_VSAG_LIB_H
#include <stdint.h>
#include <iostream>
#include <map>
#include <omp.h>

namespace ob_vsag_lib {

// // 暂时只有HNSW索引，同时操作只有创建索引，删除索引，对索引插入，对索引搜索。不支持对索引删除

int64_t example();
typedef void* VectorIndexPtr;
extern bool is_init_;
enum IndexType { INVALID_INDEX_TYPE = -1, HNSW_TYPE = 0, MAX_INDEX_TYPE };

// 吐槽，为什么vsag库里面有好几个TODO，oceanbase里面也是有几个TODO，
/**
 *   * Get the version based on git revision
 *     *
 *       * @return the version text
 *         */
extern std::string
version();

/**
 *   * Init the vsag library
 *     *
 *       * @return true always
 *         */
extern bool is_init();

/*
 * *trace = 0
 * *debug = 1
 * *info = 2
 * *warn = 3
 * *err = 4
 * *critical = 5
 * *off = 6
 * */
extern void set_log_level(int64_t level_num);
extern void set_logger(void *logger_ptr);
extern void set_block_size_limit(uint64_t size);
extern bool is_supported_index(IndexType index_type);

// 创建索引操作
extern int create_index(VectorIndexPtr& index_handler, IndexType index_type,
                        const char* dtype,
                        const char* metric,int dim,
                        int max_degree, int ef_construction, int ef_search, void* allocator);

extern int build_index(VectorIndexPtr& index_handler, float* vector_list, int64_t* ids, int dim, int size);

//
extern int add_index(VectorIndexPtr& index_handler, float* vector, int64_t* ids, int dim, int size);
extern int get_index_number(VectorIndexPtr& index_handler, int64_t &size);
extern int knn_search(VectorIndexPtr& index_handler,float* query_vector, int dim, int64_t topk,
                      const float*& dist, const int64_t*& ids, int64_t &result_size, int ef_search,
                       void* invalid);
extern int serialize(VectorIndexPtr& index_handler, const std::string dir);
extern int deserialize_bin(VectorIndexPtr& index_handler, const std::string dir);
extern int fserialize(VectorIndexPtr& index_handler, std::ostream& out_stream);
extern int fdeserialize(VectorIndexPtr& index_handler, std::istream& in_stream);
extern int delete_index(VectorIndexPtr& index_handler);
} // namesapce obvectorlib
#endif // OB_VSAG_LIB_H

