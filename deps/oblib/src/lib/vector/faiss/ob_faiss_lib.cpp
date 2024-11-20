#include "ob_faiss_lib.h"
#include "faiss/Index.h"
#include "faiss/IndexHNSW.h"
#include <fstream>
#include <chrono>

namespace obvectorlib {

void set_log_level(int64_t level_num) {
  return;
}

bool is_init() {
  return true;
}

void set_logger(void *logger_ptr) {
  return;
}

void set_block_size_limit(uint64_t size) {
  return;
}

bool is_supported_index(IndexType index_type) {
    return INVALID_INDEX_TYPE < index_type && index_type < MAX_INDEX_TYPE;
}

std::string
version() { return "woooow!!! this is DogDu's faiss in OceanBase."; }

// 创建索引操作
extern int create_index(VectorIndexPtr& index_handler, IndexType index_type,
                        const char* dtype,
                        const char* metric,int dim,
                        int max_degree, int ef_construction, int ef_search, void* allocator) {
                          return 0;
                        }

extern int build_index(VectorIndexPtr& index_handler, float* vector_list, int64_t* ids, int dim, int size) {
  return 0;
}

extern int add_index(VectorIndexPtr& index_handler, float* vector, int64_t* ids, int dim, int size) {
  return 0;
}
extern int get_index_number(VectorIndexPtr& index_handler, int64_t &size) {
  return 0;
}
extern int knn_search(VectorIndexPtr& index_handler,float* query_vector, int dim, int64_t topk,
                      const float*& dist, const int64_t*& ids, int64_t &result_size, int ef_search,
                       void* invalid) {
                        return 0;
                       }
extern int serialize(VectorIndexPtr& index_handler, const std::string dir) {
  return 0;
}
extern int deserialize_bin(VectorIndexPtr& index_handler, const std::string dir) {
  return 0;
}
extern int fserialize(VectorIndexPtr& index_handler, std::ostream& out_stream) {
  return 0;
}
extern int fdeserialize(VectorIndexPtr& index_handler, std::istream& in_stream) {
  return 0;
}
extern int delete_index(VectorIndexPtr& index_handler) {
  return 0;
}

}