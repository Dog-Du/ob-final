#include "ob_faiss_lib.h"
#include "faiss/Index.h"
#include "faiss/IndexHNSW.h"
#include "faiss/MetricType.h"
#include "faiss/impl/io.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"

#include <omp.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>

namespace obvectorlib {

enum class ErrorType {
    // start with 1, 0 is reserved

    // [common errors]
    UNKNOWN_ERROR = 1, // unknown error
    INTERNAL_ERROR,    // some internal errors occupied in algorithm
    INVALID_ARGUMENT,  // invalid argument

    // [behavior errors]
    BUILD_TWICE,     // index has been build, cannot build again
    INDEX_NOT_EMPTY, // index object is NOT empty so that should not deserialize
                     // on it
    UNSUPPORTED_INDEX,           // trying to create an unsupported index
    UNSUPPORTED_INDEX_OPERATION, // the index does not support this function
    DIMENSION_NOT_EQUAL, // the dimension of add/build/search request is NOT
                         // equal to index
    INDEX_EMPTY,         // index is empty, cannot search or serialize

    // [runtime errors]
    NO_ENOUGH_MEMORY, // failed to alloc memory
    READ_ERROR,       // cannot read from binary
    MISSING_FILE,     // some file missing in index diskann deserialization
    INVALID_BINARY,   // the content of binary is invalid
};

// 自定义流缓冲区，将 faiss::write_index 写入到 std::ostream
class StreamWriter : public faiss::IOWriter {
   public:
    explicit StreamWriter(std::ostream& os) : out_stream_(os) {}

    size_t operator()(const void* ptr, size_t size, size_t nitems) override {
        size_t total_size = size * nitems;
        out_stream_.write(static_cast<const char*>(ptr), total_size);
        return total_size;
    }

   private:
    std::ostream& out_stream_;
};

class StreamReader : public faiss::IOReader {
   public:
    explicit StreamReader(std::istream& is) : in_stream_(is) {}

    size_t operator()(void* ptr, size_t size, size_t nitems) override {
        size_t total_size = size * nitems;
        in_stream_.read(static_cast<char*>(ptr), total_size);
        return in_stream_.gcount(); // 返回实际读取的字节数
    }

   private:
    std::istream& in_stream_;
};

class HnswIndexHandler {
   public:
    HnswIndexHandler() = delete;

    HnswIndexHandler(
            bool is_create,
            bool is_build,
            bool use_static,
            int max_degree,
            int ef_construction,
            int ef_search,
            int dim,
            std::shared_ptr<faiss::Index> index)
            : is_created_(is_create),
              is_build_(is_build),
              use_static_(use_static),
              max_degree_(max_degree),
              ef_construction_(ef_construction),
              ef_search_(ef_search),
              dim_(dim),
              index_(index) {}

    ~HnswIndexHandler() {
        index_ = nullptr;
    }
    void set_build(bool is_build) {
        is_build_ = is_build;
    }
    bool is_build(bool is_build) {
        return is_build_;
    }
    // int build_index(const vsag::DatasetPtr& base);
    int get_index_number() {
        return static_cast<int>(index_->ntotal);
    }
    // int add_index(const vsag::DatasetPtr& incremental);
    // int knn_search(const vsag::DatasetPtr& query, int64_t topk,
    //               const std::string& parameters,
    //               const float*& dist, const int64_t*& ids, int64_t
    //               &result_size, const std::function<bool(int64_t)>& filter);
    std::shared_ptr<faiss::Index>& get_index() {
        return index_;
    }
    void set_index(std::shared_ptr<faiss::Index> hnsw) {
        index_ = hnsw;
    }
    inline bool get_use_static() {
        return use_static_;
    }
    inline int get_max_degree() {
        return max_degree_;
    }
    inline int get_ef_construction() {
        return ef_construction_;
    }
    inline int get_ef_search() {
        return ef_search_;
    }
    inline int get_dim() {
        return dim_;
    }

   private:
    bool is_created_;
    bool is_build_;
    bool use_static_;
    int max_degree_;
    int ef_construction_;
    int ef_search_;
    int dim_;
    std::shared_ptr<faiss::Index> index_;
};

void set_log_level(int64_t level_num) {
    return;
}

bool is_init() {
    return true;
}

void set_logger(void* logger_ptr) {
    return;
}

void set_block_size_limit(uint64_t size) {
    return;
}

bool is_supported_index(IndexType index_type) {
    return INVALID_INDEX_TYPE < index_type && index_type < MAX_INDEX_TYPE;
}

std::string version() {
    return "woooow!!! this is DogDu's faiss in OceanBase.";
}

int create_index(
        VectorIndexPtr& index_handler,
        IndexType index_type,
        const char* dtype, // useless.
        const char* metric,
        int dim,
        int max_degree,
        int ef_construction,
        int ef_search,
        void* allocator) {
    ErrorType error = ErrorType::UNKNOWN_ERROR;
    int ret = 0;
    if (dtype == nullptr || metric == nullptr) {
        return static_cast<int>(ErrorType::INVALID_ARGUMENT);
    }

    faiss::MetricType metric_type = faiss::MetricType::METRIC_Linf;
    if (strcmp(metric, "l2") == 0) {
        metric_type = faiss::METRIC_L2;
    } else if (strcmp(metric, "ip") == 0) {
        metric_type = faiss::METRIC_INNER_PRODUCT;
    } else {
        error = ErrorType::INVALID_ARGUMENT;
        return static_cast<int>(error);
    }

    bool is_support = is_supported_index(index_type);

    if (is_support) {
        omp_set_num_threads(6);
        // create index
        std::shared_ptr<faiss::Index> index;

        auto tmp_index = new faiss::IndexHNSWFlat(dim, 8, metric_type);
        tmp_index->hnsw.efConstruction = 200;
        tmp_index->hnsw.efSearch = 10;
        index.reset(tmp_index);

        HnswIndexHandler* hnsw_handler = new HnswIndexHandler(
                true,
                false,
                false,
                max_degree,
                ef_construction,
                ef_search,
                dim,
                index);

        index_handler = static_cast<VectorIndexPtr>(hnsw_handler);
        return 0;
    } else {
        error = ErrorType::UNSUPPORTED_INDEX;
    }
    ret = static_cast<int>(error);

    return ret;
}

int build_index(
        VectorIndexPtr& index_handler,
        float* vector_list,
        int64_t* ids,
        int dim,
        int size) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();

    try {
        index->add_with_ids(size, vector_list, ids);
    } catch (...) {
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    hnsw_handler->set_build(true);
    return 0;
}

int add_index(
        VectorIndexPtr& index_handler,
        float* vector,
        int64_t* ids,
        int dim,
        int size) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();

    try {
        index->add_with_ids(size, vector, ids);
    } catch (...) {
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    return 0;
}

int get_index_number(VectorIndexPtr& index_handler, int64_t& size) {
    ErrorType error = ErrorType::UNKNOWN_ERROR;
    if (index_handler == nullptr) {
        return static_cast<int>(error);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    size = hnsw->get_index_number();
    return 0;
}

int knn_search(
        VectorIndexPtr& index_handler,
        float* query_vector,
        int dim,
        int64_t topk,
        const float*& dist,
        const int64_t*& ids,
        int64_t& result_size,
        int ef_search,
        void* invalid) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();

    try {
        index->search(
                1,
                query_vector,
                topk,
                const_cast<float*&>(dist),
                const_cast<int64_t*&>(ids));
    } catch (...) {
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }

    for (int i = topk - 1; i >= 0; --i) {
        if (dist[i] != -1 && ids[i] != -1) {
            result_size = i + 1;
        }
    }

    return 0;
}

int serialize(VectorIndexPtr& index_handler, const std::string dir) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();
    std::string file_name = dir + "hnsw.data";
    try {
        faiss::write_index(index.get(), file_name.c_str());
    } catch (...) {
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    return 0;
}

extern int deserialize_bin(
        VectorIndexPtr& index_handler,
        const std::string dir) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();
    std::string file_name = dir + "hnsw.data";

    try {
        index.reset(faiss::read_index(file_name.c_str()));
    } catch (...) {
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    return 0;
}

int fserialize(VectorIndexPtr& index_handler, std::ostream& out_stream) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();
    StreamWriter writer(out_stream);

    try {
        faiss::write_index(index.get(), &writer);
    } catch (...) {
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    return 0;
}

int fdeserialize(VectorIndexPtr& index_handler, std::istream& in_stream) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();
    StreamReader reader(in_stream);

    try {
        index.reset(faiss::read_index(&reader));
    } catch (...) {
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    return 0;
}

int delete_index(VectorIndexPtr& index_handler) {
    if (index_handler != NULL) {
        delete static_cast<HnswIndexHandler*>(index_handler);
        index_handler = NULL;
    }
    return 0;
}

} // namespace obvectorlib