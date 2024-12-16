#include "ob_ngt_lib.h"

#include <NGT/Index.h>
#include <NGT/NGTQ/QuantizedGraph.h>
#include <NGT/ObjectSpace.h>
#include <omp.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "NGT/Command.h"
#include "NGT/Common.h"

#ifdef NGT_INDEX_PATH_DIR
static const std::string index_path_prex = NGT_INDEX_PATH_DIR;
#else
static const std::string index_path_prex = "./";
#endif

namespace obvectorlib {

enum class ErrorType {
    // start with 1, 0 is reserved

    // [common errors]
    UNKNOWN_ERROR = 1,  // unknown error
    INTERNAL_ERROR,     // some internal errors occupied in algorithm
    INVALID_ARGUMENT,   // invalid argument

    // [behavior errors]
    BUILD_TWICE,                  // index has been build, cannot build again
    INDEX_NOT_EMPTY,              // index object is NOT empty so that should not deserialize
                                  // on it
    UNSUPPORTED_INDEX,            // trying to create an unsupported index
    UNSUPPORTED_INDEX_OPERATION,  // the index does not support this function
    DIMENSION_NOT_EQUAL,          // the dimension of add/build/search request is NOT
                                  // equal to index
    INDEX_EMPTY,                  // index is empty, cannot search or serialize

    // [runtime errors]
    NO_ENOUGH_MEMORY,  // failed to alloc memory
    READ_ERROR,        // cannot read from binary
    MISSING_FILE,      // some file missing in index diskann deserialization
    INVALID_BINARY,    // the content of binary is invalid
};

class RowDataHandler {  // 对列数据进行简单的内存管理
public:
    explicit RowDataHandler(const char* data, uint32_t length) {
        if (length == 0) {
            data_ = nullptr;
            data_length_ = 0;
            return;
        }

        data_length_ = length;
        data_ = (char*)malloc(data_length_);

        if (data != nullptr) {  // 如果 data == nullptr，只申请空间，不进行拷贝
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

    RowDataHandler&
    operator=(RowDataHandler&& rhs) {
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

    uint32_t
    get_length() const {
        return data_length_;
    }
    char*
    data() const {
        return data_;
    }

protected:
    char* data_ = nullptr;
    uint32_t data_length_;
};

class HnswIndexHandler {
public:
    HnswIndexHandler() {
    }

    ~HnswIndexHandler() {
    }

public:
    char is_build_;
    char is_created_;
    char is_qg_;
    int32_t dim_;
    std::string index_path_;
    std::unordered_map<int64_t, int64_t> id_map_;
    std::vector<std::vector<float>> vector_list_;
    std::vector<int64_t> ids_;
    std::shared_ptr<NGT::Index> index_;
    NGTQG::Index* ngtqg_index_;
};

int64_t
example() {
    return 0;
}

void
set_log_level(int64_t level_num) {
    return;
}

bool
is_init() {
    return true;
}

void
set_logger(void* logger_ptr) {
    return;
}

void
set_block_size_limit(uint64_t size) {
    return;
}

bool
is_supported_index(IndexType index_type) {
    return INVALID_INDEX_TYPE < index_type && index_type < MAX_INDEX_TYPE;
}

std::string
version() {
    return "DogDu's NGT in OceanBase.";
}

NGTQG::Index*
get_ngtqg_index(NGT::Index* index) {
    return dynamic_cast<NGTQG::Index*>(index);
}

bool
directory_exists(const char* path) {
    struct stat info;

    if (stat(path, &info) != 0) {
        return false;
    }

    return S_ISDIR(info.st_mode);
}

int
create_index(VectorIndexPtr& index_handler,
             IndexType index_type,
             const char* dtype,  // useless.
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

    if (ret != 0) {
        return ret;
    }

    if (!is_supported_index(index_type)) {
        return static_cast<int>(ErrorType::UNSUPPORTED_INDEX);
    }

    NGT::Property property;
    property.setDefault();
    property.dimension = dim;
    property.objectType = NGT::ObjectSpace::ObjectType::Float;
    property.indexType = NGT::Property::IndexType::Graph;
    property.outgoingEdge = 64;
    property.incomingEdge = 120;
    property.edgeSizeForSearch = 100;
    property.edgeSizeForCreation = 3;
    property.threadPoolSize = 8;

    if (strcmp(metric, "l2") == 0) {
        property.distanceType = NGT::Index::Property::DistanceType::DistanceTypeL2;
    } else if (strcmp(metric, "ip") == 0) {
        property.distanceType = NGT::Index::Property::DistanceType::DistanceTypeInnerProduct;
    } else {
        return static_cast<int>(ErrorType::INVALID_ARGUMENT);
    }

    HnswIndexHandler* hnsw_handler = new HnswIndexHandler();
    std::string index_path = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::system_clock::now().time_since_epoch())
                                                .count());
    hnsw_handler->index_path_ = index_path_prex + index_path;

    if (!directory_exists(hnsw_handler->index_path_.c_str())) {
        NGT::Index::create(hnsw_handler->index_path_, property);
    }

    hnsw_handler->is_build_ = false;
    hnsw_handler->is_qg_ = false;
    hnsw_handler->is_created_ = false;
    hnsw_handler->dim_ = dim;
    hnsw_handler->index_.reset(new NGT::Index(hnsw_handler->index_path_));
    hnsw_handler->ngtqg_index_ = nullptr;
    index_handler = hnsw_handler;
    return 0;
}

int
build_index(VectorIndexPtr& index_handler, float* vector_list, int64_t* ids, int dim, int size) {
    return 1;
}

int
add_index(VectorIndexPtr& index_handler, float* vector, int64_t* ids, int dim, int size) {
    return 1;  // 这个函数不再使用。
}

int64_t
push_vector(std::vector<std::vector<float>>& vector_list,
            std::vector<int64_t>& ids,
            float* vector,
            int64_t* id,
            int size,
            int dim) {
    for (int i = 0; i < size; ++i) {
        vector_list.push_back(std::vector<float>());
        vector_list.back().reserve(dim);
        for (int j = 0; j < dim; ++j) {
            vector_list.back().push_back(vector[i * dim + j]);
        }
        ids.push_back(id[i]);
    }
    return ids.size();
}

void
build_index(HnswIndexHandler* hnsw_handler) {
    for (size_t i = 0; i < hnsw_handler->ids_.size(); ++i) {
        hnsw_handler->id_map_[hnsw_handler->index_->insert(hnsw_handler->vector_list_[i])] =
            hnsw_handler->ids_[i];
    }
    hnsw_handler->vector_list_.clear();
    hnsw_handler->ids_.clear();

    if (hnsw_handler->index_->getNumberOfObjects() >= 1000000) {
        hnsw_handler->index_->createIndex(10);
        hnsw_handler->index_->save();
        // NGTQG::Index::quantize(hnsw_handler->index_path_, 1, 96, true);
        hnsw_handler->is_build_ = true;
        // hnsw_handler->is_qg_ = true;
        // hnsw_handler->index_.reset(new NGTQG::Index(hnsw_handler->index_path_));
        // hnsw_handler->ngtqg_index_ = get_ngtqg_index(hnsw_handler->index_.get());
    }
}

int
add_index(VectorIndexPtr& index_handler,
          float* vector,
          int64_t* ids,
          int dim,
          int size,
          char* datas,
          uint32_t data_length) {
    HnswIndexHandler* hnsw_handler = static_cast<HnswIndexHandler*>(index_handler);

    if (!hnsw_handler->is_build_) {
        if (push_vector(hnsw_handler->vector_list_,
                        hnsw_handler->ids_,
                        vector,
                        ids,
                        size,
                        hnsw_handler->dim_) +
                hnsw_handler->index_->getNumberOfObjects() >=
            1000000) {
            build_index(hnsw_handler);
        }
    } else {
        std::vector<float> obj;
        for (int i = 0; i < size; ++i) {
            obj.reserve(hnsw_handler->dim_);
            for (int j = 0; j < hnsw_handler->dim_; ++j) {
                obj.push_back(vector[i * hnsw_handler->dim_ + j]);
            }

            if (hnsw_handler->is_qg_ == true) {
                hnsw_handler->id_map_[hnsw_handler->ngtqg_index_->insert(obj)] = ids[i];
            } else {
                hnsw_handler->id_map_[hnsw_handler->index_->insert(obj)] = ids[i];
            }
        }
    }

    return 0;
}

int
get_index_number(VectorIndexPtr& index_handler, int64_t& size) {
    HnswIndexHandler* hnsw_handler = static_cast<HnswIndexHandler*>(index_handler);

    size = hnsw_handler->ids_.size() + hnsw_handler->index_->getNumberOfObjects();
    return 0;
}

int
knn_search(VectorIndexPtr& index_handler,
           float* query_vector,
           int dim,
           int64_t topk,
           const float*& dist,
           const int64_t*& ids,
           int64_t& result_size,
           int ef_search,
           void* invalid) {
    return 1;
}

int
knn_search(VectorIndexPtr& index_handler,
           float* query_vector,
           int dim,
           int64_t topk,
           const float*& dist,
           const int64_t*& ids,
           int64_t& result_size,
           int ef_search,
           char**& row_datas,
           uint32_t& row_length) {
    HnswIndexHandler* hnsw_handler = static_cast<HnswIndexHandler*>(index_handler);

    if (!hnsw_handler->is_build_) {
        build_index(hnsw_handler);
    }

    std::vector<float> query;
    query.reserve(hnsw_handler->dim_);

    for (int i = 0; i < hnsw_handler->dim_; ++i) {
        query.push_back(query_vector[i]);
    }

    NGT::ObjectDistances objects;
    if (hnsw_handler->is_qg_ == true) {
        NGTQG::SearchQuery qg_sc(query);
        qg_sc.setResults(&objects);
        qg_sc.setEpsilon(0.02);
        qg_sc.setSize(topk);
        // qg_sc.radius = 1.02;
        qg_sc.edgeSize = 96;

        hnsw_handler->ngtqg_index_->search(qg_sc);
    } else {
        NGT::SearchQuery qg_sc(query);
        qg_sc.setResults(&objects);
        qg_sc.setEpsilon(0.02);
        qg_sc.setSize(topk);
        // qg_sc.radius = 1.02;
        qg_sc.edgeSize = 96;

        hnsw_handler->index_->search(qg_sc);
    }

    result_size = objects.size();
    dist = nullptr;
    ids = nullptr;
    row_length = 500;
    row_datas = nullptr;

    if (result_size > 0) {
        float* dist_res = (float*)malloc(sizeof(float) * result_size);
        int64_t* ids_res = (int64_t*)malloc(sizeof(int64_t) * result_size);
        row_datas = (char**)malloc(sizeof(char*) * result_size);

        for (int64_t i = 0; i < result_size; ++i) {
            dist_res[i] = objects[i].distance;
            ids_res[i] = hnsw_handler->id_map_[objects[i].id];
        }

        dist = dist_res;
        ids = ids_res;
    }
    return 0;
}

int
serialize(VectorIndexPtr& index_handler, const std::string dir) {
    std::string file_name = dir + "hnsw.data";
    std::ofstream ftream(file_name, std::ios_base::out);
    return fserialize(index_handler, ftream);
}

int
deserialize_bin(VectorIndexPtr& index_handler, const std::string dir) {
    std::string file_name = dir + "hnsw.data";
    std::ifstream fstream(file_name, std::ios_base::in);
    return fdeserialize(index_handler, fstream);
}

void
save_id_map(std::unordered_map<int64_t, int64_t>& id_map, std::ostream& out_stream) {
    size_t size = id_map.size();
    out_stream.write((char*)&size, sizeof(size));

    for (auto& kv : id_map) {
        out_stream.write((char*)&kv.first, sizeof(kv.first));
        out_stream.write((char*)&kv.second, sizeof(kv.second));
    }
}

int
fserialize(VectorIndexPtr& index_handler, std::ostream& out_stream) {
    HnswIndexHandler* hnsw_handler = static_cast<HnswIndexHandler*>(index_handler);
    if (!hnsw_handler->is_build_) {
        build_index(hnsw_handler);
    }

    size_t size = hnsw_handler->index_path_.size();
    out_stream.write((char*)&size, sizeof(size));
    out_stream.write(hnsw_handler->index_path_.c_str(), hnsw_handler->index_path_.size());
    out_stream.write((char*)&hnsw_handler->is_qg_, sizeof(hnsw_handler->is_qg_));        // 1.qg
    out_stream.write((char*)&hnsw_handler->is_build_, sizeof(hnsw_handler->is_build_));  // 2.build
    out_stream.write((char*)&hnsw_handler->is_created_,
                     sizeof(hnsw_handler->is_created_));  // 3.create
    save_id_map(hnsw_handler->id_map_, out_stream);
    return 0;
}

void
read_id_map(std::unordered_map<int64_t, int64_t>& id_map, std::istream& in_stream) {
    size_t size = 0;
    int64_t k;
    int64_t v;
    in_stream.read((char*)&size, sizeof(size));

    for (size_t i = 0; i < size; ++i) {
        in_stream.read((char*)&k, sizeof(k));
        in_stream.read((char*)&v, sizeof(v));
        id_map[k] = v;
    }
}

int
fdeserialize(VectorIndexPtr& index_handler, std::istream& in_stream) {
    HnswIndexHandler* hnsw_handler = static_cast<HnswIndexHandler*>(index_handler);
    if (hnsw_handler != nullptr) {
        delete hnsw_handler;
        hnsw_handler = new HnswIndexHandler();
        index_handler = static_cast<VectorIndexPtr>(hnsw_handler);
    }

    size_t size;
    in_stream.read((char*)&size, sizeof(size));
    hnsw_handler->index_path_.resize(size);
    in_stream.read((char*)hnsw_handler->index_path_.c_str(), size);
    in_stream.read((char*)&hnsw_handler->is_qg_, sizeof(hnsw_handler->is_qg_));        // 1.qg
    in_stream.read((char*)&hnsw_handler->is_build_, sizeof(hnsw_handler->is_build_));  // 2.build
    in_stream.read((char*)&hnsw_handler->is_created_,
                   sizeof(hnsw_handler->is_created_));  // 3.create

    read_id_map(hnsw_handler->id_map_, in_stream);

    if (hnsw_handler->is_qg_ == true) {
        hnsw_handler->ngtqg_index_ = new NGTQG::Index(hnsw_handler->index_path_);
        hnsw_handler->index_.reset(hnsw_handler->ngtqg_index_);
    } else {
        hnsw_handler->ngtqg_index_ = nullptr;
        hnsw_handler->index_.reset(new NGT::Index(hnsw_handler->index_path_));
    }

    NGT::Property pp;
    hnsw_handler->index_->getProperty(pp);
    hnsw_handler->dim_ = pp.dimension;
    return 0;
}

int
delete_index(VectorIndexPtr& index_handler) {
    if (index_handler != NULL) {
        delete static_cast<HnswIndexHandler*>(index_handler);
        index_handler = NULL;
    }
    return 0;
}
}  // namespace obvectorlib