#include "faiss/ob_faiss_lib.h"
#include <cstdint>
#include <iostream>
#include <ostream>
#include <sstream>

int main() {
  std::cout << obvectorlib::version() << std::endl;
  obvectorlib::VectorIndexPtr index_ptr = nullptr;
  std::cout << obvectorlib::create_index(index_ptr,
                                         obvectorlib::IndexType::HNSW_TYPE,
                                         "float32", "l2", 3, 10, 10, 10)
            << std::endl;
  float dataset[][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  std::cout << "serialize" << std::endl;
  std::cout << obvectorlib::serialize(index_ptr, "") << std::endl;
  std::cout << obvectorlib::build_index(index_ptr, (float *)dataset, nullptr, 3,
                                        3)
            << std::endl;
  std::ostringstream s;
  obvectorlib::fserialize(index_ptr, s);
  obvectorlib::deserialize_bin(index_ptr, "const std::string dir");
  int64_t x;
  obvectorlib::add_index(index_ptr, nullptr, nullptr, 0, 0);
  obvectorlib::knn_search(index_ptr, nullptr, 0, 0, (const float *&)dataset,
                          (const int64_t *&)dataset, x, 0);
  obvectorlib::IndexType y;
  obvectorlib::set_logger(nullptr);
  obvectorlib::example();
  obvectorlib::is_init();
  obvectorlib::deserialize_bin(index_ptr, "");
  obvectorlib::delete_index(index_ptr);
  std::istringstream ss;
  obvectorlib::fdeserialize(index_ptr, ss);
  obvectorlib::get_index_number(index_ptr, x);
  obvectorlib::set_block_size_limit(0);
  obvectorlib::set_log_level(0);
  obvectorlib::is_supported_index(obvectorlib::IndexType::INVALID_INDEX_TYPE);
  return 0;
}