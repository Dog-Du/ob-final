#include "ob_faiss_lib.h"
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/MetricType.h>
#include <faiss/impl/io.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <linux/limits.h>
#include <omp.h>

#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <unordered_map>
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/FaissException.h"
#include "faiss/impl/HNSW.h"

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
        return nitems;
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
        return in_stream_.gcount() / size;
    }

   private:
    std::istream& in_stream_;
};

class RowDataHandler { // 对列数据进行简单的内存管理
   public:
    explicit RowDataHandler(const char* data, uint32_t length) {
        if (length == 0) {
            data_ = nullptr;
            data_length_ = 0;
            return;
        }

        data_length_ = length;
        data_ = (char*)malloc(data_length_);

        if (data != nullptr) { // 如果 data == nullptr，只申请空间，不进行拷贝
            memcpy(data_, data, data_length_);
        }
    }

    ~RowDataHandler() {
        if (data_ != nullptr) {
            free(data_);
        }
        data_ = nullptr;
        data_length_ = 0;
    }

    RowDataHandler(const RowDataHandler& rhs) = delete;

    RowDataHandler& operator=(RowDataHandler&& rhs) {
        if (this == &rhs) {
            return *this;
        }

        this->~RowDataHandler();
        data_ = rhs.data_;
        data_length_ = rhs.data_length_;
        rhs.data_ = nullptr;
        rhs.data_length_ = 0;
        return *this;
    }

    RowDataHandler(RowDataHandler&& rhs) {
        data_ = rhs.data_;
        data_length_ = rhs.data_length_;
        rhs.data_ = nullptr;
        rhs.data_length_ = 0;
    }

    uint32_t get_length() const {
        return data_length_;
    }
    char* data() const {
        return data_;
    }

   protected:
    char* data_ = nullptr;
    uint32_t data_length_;
};

class VectorDataHandler : public RowDataHandler {

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
            std::shared_ptr<faiss::Index> index,
            std::shared_ptr<faiss::IndexIDMap> ix_id_map,
            std::shared_ptr<faiss::IndexFlatL2> quantizer)
            : is_created_(is_create),
              is_build_(is_build),
              use_static_(use_static),
              max_degree_(max_degree),
              ef_construction_(ef_construction),
              ef_search_(ef_search),
              dim_(dim),
              index_(index),
              ix_id_map_(ix_id_map),
              quantizer_(quantizer) {}

    ~HnswIndexHandler() {
        index_ = nullptr;
    }
    void set_build(bool is_build) {
        is_build_ = is_build;
    }
    bool is_build() {
        return is_build_;
    }
    // int build_index(const vsag::DatasetPtr& base);
    inline int get_index_number() {
        return static_cast<int>(ix_id_map_->ntotal);
    }
    // int add_index(const vsag::DatasetPtr& incremental);
    // int knn_search(const vsag::DatasetPtr& query, int64_t topk,
    //               const std::string& parameters,
    //               const float*& dist, const int64_t*& ids, int64_t
    //               &result_size, const std::function<bool(int64_t)>& filter);
    inline std::shared_ptr<faiss::IndexIDMap>& get_index() {
        return ix_id_map_;
    }
    // inline std::shared_ptr<faiss::Index>& get_index() {
    //     return index_;
    // }
    inline void set_index(std::shared_ptr<faiss::Index> hnsw) {
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

   public:
    bool is_created_;
    bool is_build_;
    bool use_static_;
    int max_degree_;
    int ef_construction_;
    int ef_search_;
    int dim_;
    std::vector<float> vector_list_;
    std::vector<int64_t> ids_;
    std::shared_ptr<faiss::Index> index_;
    std::shared_ptr<faiss::IndexIDMap> ix_id_map_;
    std::shared_ptr<faiss::IndexFlatL2> quantizer_;

    // std::unordered_map<int64_t, RowDataHandler> id_data_map_;
};

bool& get_init() {
    static bool init = false;
    return init;
}

// std::vector<float>& get_static_vector_list() {
//     static std::vector<float> vector_list;
//     return vector_list;
// }

// std::vector<int64_t>& get_static_ids() {
//     static std::vector<int64_t> ids;
//     return ids;
// }

int64_t example() {
    return 0;
}

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
    int ret = 0;
    if (dtype == nullptr || metric == nullptr) {
        return static_cast<int>(ErrorType::INVALID_ARGUMENT);
    }

    // printf("[FAISS][DEBUG] create_index ::: dtype : %s, metric : %s, dim :
    // %d\n",
    //        dtype,
    //        metric,
    //        dim);
    faiss::MetricType metric_type = faiss::MetricType::METRIC_Linf;

    if (strcmp(metric, "l2") == 0) {
        metric_type = faiss::METRIC_L2;
    } else if (strcmp(metric, "ip") == 0) {
        metric_type = faiss::METRIC_INNER_PRODUCT;
    } else {
        ret = static_cast<int>(ErrorType::INVALID_ARGUMENT);
    }

    if (ret != 0) {
        return ret;
    }

    bool is_support = is_supported_index(index_type);

    if (is_support) {
        omp_set_num_threads(10);
        // create index
        std::shared_ptr<faiss::Index> index;
        std::shared_ptr<faiss::IndexIDMap> ix_id_map;
        std::shared_ptr<faiss::IndexFlatL2> quantizer;

        // ann-benchmarks上的召回率0.98331，pqs=2224，M=16，efConstruction=500,efSearch=80
        auto tmp_index = new faiss::IndexHNSWFlat(dim, 24, metric_type);
        tmp_index->hnsw.efConstruction = 1000; // 提升一点，反正构建时间足够。
        tmp_index->hnsw.efSearch = 80;
        index.reset(tmp_index);

        ix_id_map.reset(new faiss::IndexIDMap(index.get()));

        HnswIndexHandler* hnsw_handler = new HnswIndexHandler(
                true,
                false,
                false,
                max_degree,
                ef_construction,
                6000,
                dim,
                index,
                ix_id_map,
                quantizer);

        // get_static_vector_list().reserve(128LL * 1000000);
        // get_static_ids().reserve(1000000);
        index_handler = static_cast<VectorIndexPtr>(hnsw_handler);
    } else {
        ret = static_cast<int>(ErrorType::UNSUPPORTED_INDEX);
    }

    return ret;
}

int build_index(
        VectorIndexPtr& index_handler,
        float* vector_list,
        int64_t* ids,
        int dim,
        int size) {
    return 1;
    // printf("[FAISS][DEBUG] create_index ::: dim : %d, size : %d\n", dim,
    // size);
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();

    int ret = 0;
    try {
        if (!index->is_trained) {
            index->train(size, vector_list);
            index->is_trained = true;
        }
        index->add_with_ids(size, vector_list, ids);
    } catch (...) {
        ret = static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    hnsw_handler->set_build(true);
    return ret;
}

int add_index(
        VectorIndexPtr& index_handler,
        float* vector,
        int64_t* ids,
        int dim,
        int size) {
    return 1; // 这个函数不再使用。

    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();

    if (!get_init()) {
        for (int64_t i = 0, n = 1LL * size * dim; i < n; ++i) {
            hnsw_handler->vector_list_.push_back(vector[i]);
        }

        for (int64_t i = 0; i < size; ++i) {
            hnsw_handler->ids_.push_back(ids[i]);
        }

        if (hnsw_handler->ids_.size() >= 1000'000) {
            FAISS_ASSERT(
                    hnsw_handler->ids_.size() * hnsw_handler->get_dim() ==
                    hnsw_handler->vector_list_.size());

            try {
                if (!index->is_trained) {
                    index->train(
                            hnsw_handler->ids_.size(),
                            hnsw_handler->vector_list_.data());
                }

                index->add_with_ids(
                        hnsw_handler->ids_.size(),
                        hnsw_handler->vector_list_.data(),
                        hnsw_handler->ids_.data());
            } catch (faiss::FaissException& e) {
                std::cout << e.what() << std::endl;
                return static_cast<int>(ErrorType::UNKNOWN_ERROR);
            }
            hnsw_handler->ids_.clear();
            hnsw_handler->vector_list_.clear();
            // omp_set_num_threads(8);
            FAISS_ASSERT(index->ntotal >= 1000'000);
            get_init() = true;
        }

        return 0;
    }

    int ret = 0;
    try {
        index->add_with_ids(size, vector, ids);
    } catch (...) {
        ret = static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    return ret;
}

int add_index(
        VectorIndexPtr& index_handler,
        float* vector,
        int64_t* ids,
        int dim,
        int size,
        char* datas,
        uint32_t data_length) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();

    // for (int i = 0, offset = 0; i < size; ++i) {
    //     RowDataHandler handler(datas + offset, data_length);
    //     hnsw_handler->id_data_map_.emplace(ids[i], std::move(handler));
    //     offset += data_length;
    // }

    if (!get_init()) {
        for (int64_t i = 0, n = 1LL * size * dim; i < n; ++i) {
            hnsw_handler->vector_list_.push_back(vector[i]);
        }

        for (int64_t i = 0; i < size; ++i) {
            hnsw_handler->ids_.push_back(ids[i]);
        }

        if (hnsw_handler->ids_.size() >= 1000'000) {
            FAISS_ASSERT(
                    hnsw_handler->ids_.size() * hnsw_handler->get_dim() ==
                    hnsw_handler->vector_list_.size());

            try {
                if (!index->is_trained) {
                    index->train(
                            hnsw_handler->ids_.size(),
                            hnsw_handler->vector_list_.data());
                }

                index->add_with_ids(
                        hnsw_handler->ids_.size(),
                        hnsw_handler->vector_list_.data(),
                        hnsw_handler->ids_.data());
            } catch (faiss::FaissException& e) {
                std::cout << e.what() << std::endl;
                return static_cast<int>(ErrorType::UNKNOWN_ERROR);
            }
            hnsw_handler->ids_.clear();
            hnsw_handler->vector_list_.clear();
            // omp_set_num_threads(8);
            FAISS_ASSERT(index->ntotal >= 1000'000);
            get_init() = true;
            // FAISS_ASSERT(hnsw_handler->id_data_map_.size() == index->ntotal);
        }

        return 0;
    }

    int ret = 0;
    try {
        index->add_with_ids(size, vector, ids);
    } catch (...) {
        ret = static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    return ret;
}

int get_index_number(VectorIndexPtr& index_handler, int64_t& size) {
    if (index_handler == nullptr) {
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    HnswIndexHandler* hnsw = static_cast<HnswIndexHandler*>(index_handler);
    size = hnsw->get_index_number() + hnsw->ids_.size();
    return 0;
}

void set_hnsw_efsearch(faiss::Index* i, int topk, int ef_search) {
    if (auto x = dynamic_cast<faiss::IndexIDMap*>(i)) {
        set_hnsw_efsearch(x->index, topk, ef_search);
    } else if (auto x = dynamic_cast<faiss::IndexHNSW*>(i)) {
        x->hnsw.efSearch = topk >= 10000 ? ef_search : 80;
    } else {
        FAISS_ASSERT(false);
    }
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

    // 使用malloc而不是使用new的原因：在适配层，会给与一个内存池分配，他会进行内存释放。
    // 但是这里的内存是自己使用的，为了防止内存泄漏，使用malloc和free，而不是new和delete
    float* dist_result = (float*)malloc(sizeof(float) * topk);
    int64_t* ids_result = (int64_t*)malloc(sizeof(int64_t) * topk);

    if (!hnsw_handler->ids_.empty()) {
        FAISS_ASSERT(
                hnsw_handler->ids_.size() * hnsw_handler->get_dim() ==
                hnsw_handler->vector_list_.size());

        try {
            if (!index->is_trained) {
                index->train(
                        hnsw_handler->ids_.size(),
                        hnsw_handler->vector_list_.data());
            }
            index->add_with_ids(
                    hnsw_handler->ids_.size(),
                    hnsw_handler->vector_list_.data(),
                    hnsw_handler->ids_.data());
        } catch (faiss::FaissException& e) {
            std::cout << e.what() << std::endl;
            return static_cast<int>(ErrorType::UNKNOWN_ERROR);
        }
        hnsw_handler->ids_.clear();
        hnsw_handler->vector_list_.clear();
        // FAISS_ASSERT(hnsw_handler->id_data_map_.size() == index->ntotal);
    }

    // FAISS_ASSERT(hnsw_handler->id_data_map_.size() == index->ntotal);
    set_hnsw_efsearch(index.get(), topk, 6000);
    int ret = 0;
    try {
        index->search(1, query_vector, topk, dist_result, ids_result);
    } catch (faiss::FaissException& e) {
        std::cout << e.what() << std::endl;
        free(dist_result);
        free(ids_result);
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }

    result_size = 0;
    for (int64_t i = topk - 1; i >= 0; --i) {
        if (ids_result[i] != -1) {
            result_size = i + 1;
            break;
        }
    }

    dist = dist_result;
    ids = ids_result;
    return ret;
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
        char**& row_datas,
        uint32_t& row_length) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();

    // 使用malloc而不是使用new的原因：在适配层，会给与一个内存池分配，他会进行内存释放。
    // 但是这里的内存是自己使用的，为了防止内存泄漏，使用malloc和free，而不是new和delete
    float* dist_result = (float*)malloc(sizeof(float) * topk);
    int64_t* ids_result = (int64_t*)malloc(sizeof(int64_t) * topk);

    if (!hnsw_handler->ids_.empty()) {
        FAISS_ASSERT(
                hnsw_handler->ids_.size() * hnsw_handler->get_dim() ==
                hnsw_handler->vector_list_.size());

        try {
            if (!index->is_trained) {
                index->train(
                        hnsw_handler->ids_.size(),
                        hnsw_handler->vector_list_.data());
            }
            index->add_with_ids(
                    hnsw_handler->ids_.size(),
                    hnsw_handler->vector_list_.data(),
                    hnsw_handler->ids_.data());
        } catch (faiss::FaissException& e) {
            std::cout << e.what() << std::endl;
            return static_cast<int>(ErrorType::UNKNOWN_ERROR);
        }
        hnsw_handler->ids_.clear();
        hnsw_handler->vector_list_.clear();
        // FAISS_ASSERT(hnsw_handler->id_data_map_.size() == index->ntotal);
    }

    set_hnsw_efsearch(index.get(), topk, 6000);
    // 不知道为什么如果没有加FAISS_TEST就会产生空表。
    // FAISS_ASSERT(index->ntotal == hnsw_handler->id_data_map_.size());
    int ret = 0;
    try {
        index->search(1, query_vector, topk, dist_result, ids_result);
    } catch (faiss::FaissException& e) {
        std::cout << e.what() << std::endl;
        free(dist_result);
        free(ids_result);
        return static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }

    result_size = 0;
    for (int64_t i = topk - 1; i >= 0; --i) {
        if (ids_result[i] != -1) {
            result_size = i + 1;
            break;
        }
    }

    dist = dist_result;
    ids = ids_result;

    row_datas = nullptr;
    row_length = 0;

    if (result_size > 0) {
        // auto iter = hnsw_handler->id_data_map_.begin();
        // row_length = iter->second.get_length();
        row_length = 500;
        row_datas = (char**)malloc(result_size * sizeof(char*));
        // for (int64_t i = 0; i < result_size; ++i) {
        //     iter = hnsw_handler->id_data_map_.find(ids[i]);
        //     row_datas[i] = iter->second.data();
        // }
    }
    return ret;
}

void serialize_data_map(
        std::unordered_map<int64_t, RowDataHandler>& map,
        StreamWriter& writer) {
    if (map.empty()) {
        uint64_t size = map.size();
        uint32_t length = 0;
        FAISS_ASSERT(writer(&size, sizeof(size), 1) == 1);
        return;
    }

    uint64_t size = map.size();
    int64_t id = 0;
    uint32_t row_length = map.begin()->second.get_length();
    uint32_t length = row_length + sizeof(id);

    RowDataHandler all_data(
            nullptr, size * length + sizeof(size) + sizeof(length));

    int64_t offset = 0;
    memcpy(all_data.data() + offset, &size, sizeof(size));
    offset += sizeof(size);
    memcpy(all_data.data() + offset, &length, sizeof(length));
    offset += sizeof(length);

    for (auto& kv : map) {
        memcpy(all_data.data() + offset, &kv.first, sizeof(kv.first));
        offset += sizeof(kv.first);
        memcpy(all_data.data() + offset, kv.second.data(), row_length);
        offset += row_length;
    }

    FAISS_ASSERT(offset == all_data.get_length());
    FAISS_ASSERT(writer(all_data.data(), all_data.get_length(), 1) == 1);
}

void deserialize_data_map(
        std::unordered_map<int64_t, RowDataHandler>& map,
        StreamReader& reader) {
    uint64_t size = 0;
    int64_t id = 0;
    uint32_t length = 0;
    uint32_t row_length = 0;

    FAISS_ASSERT(reader(&size, sizeof(size), 1) == 1);

    if (size == 0) {
        return;
    }

    FAISS_ASSERT(reader(&length, sizeof(length), 1) == 1);
    row_length = length - sizeof(id);

    RowDataHandler all_data(nullptr, length * size);
    FAISS_ASSERT(reader(all_data.data(), all_data.get_length(), 1) == 1);
    int64_t offset = 0;

    for (int64_t i = 0; i < size; ++i) {
        memcpy(&id, all_data.data() + offset, sizeof(id));
        offset += sizeof(id);
        RowDataHandler row_handler(all_data.data() + offset, row_length);
        offset += row_length;
        map.emplace(id, std::move(row_handler));
    }
}

int serialize(VectorIndexPtr& index_handler, const std::string dir) {
    std::string file_name = dir + "hnsw.data";
    std::ofstream ftream(file_name, std::ios_base::out);
    return fserialize(index_handler, ftream);
}

int deserialize_bin(VectorIndexPtr& index_handler, const std::string dir) {
    std::string file_name = dir + "hnsw.data";
    std::ifstream fstream(file_name, std::ios_base::in);
    return fdeserialize(index_handler, fstream);
}

int fserialize(VectorIndexPtr& index_handler, std::ostream& out_stream) {
    HnswIndexHandler* hnsw_handler =
            static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();

    if (!hnsw_handler->ids_.empty()) {
        FAISS_ASSERT(
                hnsw_handler->ids_.size() * hnsw_handler->get_dim() ==
                hnsw_handler->vector_list_.size());

        try {
            if (!index->is_trained) {
                index->train(
                        hnsw_handler->ids_.size(),
                        hnsw_handler->vector_list_.data());
            }
            index->add_with_ids(
                    hnsw_handler->ids_.size(),
                    hnsw_handler->vector_list_.data(),
                    hnsw_handler->ids_.data());
        } catch (faiss::FaissException& e) {
            std::cout << e.what() << std::endl;
            return static_cast<int>(ErrorType::UNKNOWN_ERROR);
        }
        hnsw_handler->ids_.clear();
        hnsw_handler->vector_list_.clear();
        // omp_set_num_threads(8);
    }

    // FAISS_ASSERT(hnsw_handler->id_data_map_.size() == index->ntotal);
    StreamWriter writer(out_stream);
    int ret = 0;
    try {
        // serialize_data_map(hnsw_handler->id_data_map_, writer);
        faiss::write_index(index.get(), &writer);
    } catch (faiss::FaissException& e) {
        std::cout << e.what() << std::endl;
        ret = static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }
    return ret;
}

int fdeserialize(VectorIndexPtr& index_handler, std::istream& in_stream) {
    omp_set_num_threads(10);
    auto* hnsw_handler = static_cast<HnswIndexHandler*>(index_handler);
    bool use_static = hnsw_handler->get_use_static();
    int max_degree = hnsw_handler->get_max_degree();
    int ef_construction = hnsw_handler->get_ef_construction();
    int ef_search = hnsw_handler->get_ef_search();
    int dim = hnsw_handler->get_dim();
    int ret = 0;
    delete_index(index_handler);
    if ((ret = create_index(
                 index_handler,
                 IndexType::HNSW_TYPE,
                 "float32",
                 "l2", // 默认写成l2
                 dim,
                 max_degree,
                 ef_construction,
                 ef_search, nullptr)) != 0) {
        return ret;
    }

    hnsw_handler = static_cast<HnswIndexHandler*>(index_handler);
    auto& index = hnsw_handler->get_index();

    StreamReader reader(in_stream);
    try {
        // deserialize_data_map(hnsw_handler->id_data_map_, reader);
        faiss::Index* i = faiss::read_index(&reader);

        if (auto x = dynamic_cast<faiss::IndexIDMap*>(i)) {
            index.reset(x);
        } else {
            index.reset(new faiss::IndexIDMap(i));
        }
    } catch (faiss::FaissException& e) {
        std::cout << e.what() << std::endl;
        ret = static_cast<int>(ErrorType::UNKNOWN_ERROR);
    }

    // FAISS_ASSERT(hnsw_handler->id_data_map_.size() == index->ntotal);
    return ret;
}

int delete_index(VectorIndexPtr& index_handler) {
    if (index_handler != NULL) {
        delete static_cast<HnswIndexHandler*>(index_handler);
        index_handler = NULL;
    }
    return 0;
}

} // namespace obvectorlib