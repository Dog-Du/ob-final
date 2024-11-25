/**
 * Copyright (c) 2021 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PubL v2. You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PubL v2 for more details.
 */

#ifndef OB_VECTOR_UTIL_H
#define OB_VECTOR_UTIL_H
#include <stdint.h>
// #include <vsag/ob_vsag_lib.h>
#include <vsag/allocator.h>
#include <vsag/logger.h>
#include <fstream>
#include <omp.h>

#include "faiss/ob_faiss_lib.h"

namespace oceanbase {
namespace common {
namespace obvectorutil {

enum Level : int {
  kTRACE = 0,
  kDEBUG = 1,
  kINFO = 2,
  kWARN = 3,
  kERR = 4,
  kCRITICAL = 5,
  kOFF = 6,
  kN_LEVELS
};

class ObVsagLogger {
public:
  void SetLevel(Level Log_level);
  void Trace(const std::string &msg);

  void Debug(const std::string &msg);

  void Info(const std::string &msg);

  void Warn(const std::string &msg);

  void Error(const std::string &msg);

  void Critical(const std::string &msg);
};

int init_vasg_logger(void *logger);

bool check_vsag_init();

int create_index(obvectorlib::VectorIndexPtr &index_handler, int index_type,
                 const char *dtype, const char *metric, int dim, int max_degree,
                 int ef_construction, int ef_search, void *allocator = NULL);

int build_index(obvectorlib::VectorIndexPtr index_handler, float *vector_list,
                int64_t *ids, int dim, int size);

int add_index(obvectorlib::VectorIndexPtr index_handler, float *vector_list,
              int64_t *ids, int dim, int size);

int get_index_number(obvectorlib::VectorIndexPtr index_handler, int64_t &size);

int knn_search(obvectorlib::VectorIndexPtr index_handler, float *query_vector,
               int dim, int64_t topk, const float *&result_dist,
               const int64_t *&result_ids, int64_t &result_size, int ef_search,
               void *invalid = NULL);

int fserialize(obvectorlib::VectorIndexPtr index_handler,
               std::ostream &out_stream);

int fdeserialize(obvectorlib::VectorIndexPtr &index_handler,
                 std::istream &in_stream);
int delete_index(obvectorlib::VectorIndexPtr &index_handler);
} // namespace obvectorutil
} // namespace common
} // namespace oceanbase
#endif /* OB_VECTOR_UTIL_H */
