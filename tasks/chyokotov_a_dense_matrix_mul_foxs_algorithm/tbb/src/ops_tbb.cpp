#include "chyokotov_a_dense_matrix_mul_foxs_algorithm/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "chyokotov_a_dense_matrix_mul_foxs_algorithm/common/include/common.hpp"

namespace chyokotov_a_dense_matrix_mul_foxs_algorithm {

ChyokotovADenseMatMulFoxAlgorithmTBB::ChyokotovADenseMatMulFoxAlgorithmTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ChyokotovADenseMatMulFoxAlgorithmTBB::ValidationImpl() {
  return (GetInput().first.size() == GetInput().second.size());
}

bool ChyokotovADenseMatMulFoxAlgorithmTBB::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(GetInput().first.size(), 0.0);
  return true;
}

int ChyokotovADenseMatMulFoxAlgorithmTBB::CalculateBlockSize(int n) {
  return static_cast<int>(std::sqrt(static_cast<double>(n)));
}

int ChyokotovADenseMatMulFoxAlgorithmTBB::CountBlock(int n, int size) {
  return (n + size - 1) / size;
}

bool ChyokotovADenseMatMulFoxAlgorithmTBB::RunImpl() {
  std::vector<double> a = GetInput().first;
  std::vector<double> b = GetInput().second;
  int n = static_cast<int>(std::sqrt(static_cast<double>(a.size())));
  if (n == 0) {
    return true;
  }

  int block_size = CalculateBlockSize(n);
  int count_block = CountBlock(n, block_size);

  tbb::parallel_for(tbb::blocked_range2d<int>(0, count_block, 0, count_block), [&](const tbb::blocked_range2d<int> &r) {
    for (int ic = r.rows().begin(); ic < r.rows().end(); ++ic) {
      for (int jc = r.cols().begin(); jc < r.cols().end(); ++jc) {
        for (int kc = 0; kc < count_block; ++kc) {
          int istart = ic * block_size;
          int jstart = jc * block_size;
          int kstart = kc * block_size;

          int iend = std::min(istart + block_size, n);
          int jend = std::min(jstart + block_size, n);
          int kend = std::min(kstart + block_size, n);

          tbb::parallel_for(istart, iend, [&](int i) {
            double *output_row = GetOutput().data() + i * n;
            const double *a_row = a.data() + i * n;

            for (int j = jstart; j < jend; ++j) {
              double sum = 0.0;
              const double *b_col = b.data() + j;
              for (int k = kstart; k < kend; ++k) {
                sum += a_row[k] * b_col[k * n];
              }
              output_row[j] += sum;
            }
          });
        }
      }
    }
  });

  return true;
}

bool ChyokotovADenseMatMulFoxAlgorithmTBB::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_a_dense_matrix_mul_foxs_algorithm
