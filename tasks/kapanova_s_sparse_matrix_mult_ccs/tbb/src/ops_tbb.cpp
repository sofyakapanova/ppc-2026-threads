#include "kapanova_s_sparse_matrix_mult_ccs/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kapanova_s_sparse_matrix_mult_ccs/common/include/common.hpp"

namespace kapanova_s_sparse_matrix_mult_ccs {

KapanovaSSparseMatrixMultCCSTBB::KapanovaSSparseMatrixMultCCSTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSSparseMatrixMultCCSTBB::ValidationImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  if (a.cols != b.rows) {
    return false;
  }
  if (a.rows == 0 || a.cols == 0 || b.rows == 0 || b.cols == 0) {
    return false;
  }
  if (a.col_ptrs.size() != static_cast<size_t>(a.cols + 1)) {
    return false;
  }
  if (b.col_ptrs.size() != static_cast<size_t>(b.cols + 1)) {
    return false;
  }
  return true;
}

bool KapanovaSSparseMatrixMultCCSTBB::PreProcessingImpl() {
  return true;
}
bool KapanovaSSparseMatrixMultCCSTBB::PostProcessingImpl() {
  return true;
}

namespace {

void CountColumnSizes(const CCSMatrix &a, const CCSMatrix &b, tbb::combinable<std::vector<size_t>> &comb_sizes,
                      const tbb::blocked_range<size_t> &range) {
  auto &sizes = comb_sizes.local();
  std::vector<double> accum(a.rows, 0.0);
  std::vector<char> mask(a.rows, 0);
  std::vector<size_t> active;
  active.reserve(a.rows / 10);

  for (size_t j = range.begin(); j != range.end(); ++j) {
    for (size_t k = b.col_ptrs[j]; k < b.col_ptrs[j + 1]; ++k) {
      size_t row_b = b.row_indices[k];
      double val_b = b.values[k];
      for (size_t zc = a.col_ptrs[row_b]; zc < a.col_ptrs[row_b + 1]; ++zc) {
        size_t i = a.row_indices[zc];
        double val_a = a.values[zc];
        if (mask[i] == 0) {
          mask[i] = 1;
          active.push_back(i);
          accum[i] = val_a * val_b;
        } else {
          accum[i] += val_a * val_b;
        }
      }
    }
    for (size_t i : active) {
      if (accum[i] != 0.0) {
        sizes[j]++;
      }
      mask[i] = 0;
      accum[i] = 0.0;
    }
    active.clear();
  }
}

void FillColumnValues(const CCSMatrix &a, const CCSMatrix &b, OutType &c,
                      tbb::combinable<std::vector<size_t>> &comb_pos, const tbb::blocked_range<size_t> &range) {
  auto &pos = comb_pos.local();
  std::vector<double> accum(a.rows, 0.0);
  std::vector<char> mask(a.rows, 0);
  std::vector<size_t> active;
  active.reserve(a.rows / 10);

  for (size_t j = range.begin(); j != range.end(); ++j) {
    for (size_t k = b.col_ptrs[j]; k < b.col_ptrs[j + 1]; ++k) {
      size_t row_b = b.row_indices[k];
      double val_b = b.values[k];
      for (size_t zc = a.col_ptrs[row_b]; zc < a.col_ptrs[row_b + 1]; ++zc) {
        size_t i = a.row_indices[zc];
        double val_a = a.values[zc];
        if (mask[i] == 0) {
          mask[i] = 1;
          active.push_back(i);
          accum[i] = val_a * val_b;
        } else {
          accum[i] += val_a * val_b;
        }
      }
    }
    std::ranges::sort(active);
    for (size_t i : active) {
      if (accum[i] != 0.0) {
        c.row_indices[pos[j]] = i;
        c.values[pos[j]] = accum[i];
        pos[j]++;
      }
      mask[i] = 0;
      accum[i] = 0.0;
    }
    active.clear();
  }
}

}  // namespace

bool KapanovaSSparseMatrixMultCCSTBB::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  auto &c = GetOutput();

  c.rows = a.rows;
  c.cols = b.cols;

  const size_t cols_sz = c.cols;

  tbb::combinable<std::vector<size_t>> comb_sizes([cols_sz] { return std::vector<size_t>(cols_sz, 0); });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, cols_sz, 64),
                    [&](const tbb::blocked_range<size_t> &range) { CountColumnSizes(a, b, comb_sizes, range); });

  std::vector<size_t> col_sizes(cols_sz, 0);
  comb_sizes.combine_each([&](const std::vector<size_t> &local) {
    for (size_t j = 0; j < cols_sz; ++j) {
      col_sizes[j] += local[j];
    }
  });

  c.col_ptrs.resize(cols_sz + 1);
  size_t offset = 0;
  for (size_t j = 0; j < cols_sz; ++j) {
    c.col_ptrs[j] = offset;
    offset += col_sizes[j];
  }
  c.col_ptrs[cols_sz] = offset;
  c.nnz = offset;
  c.values.resize(c.nnz);
  c.row_indices.resize(c.nnz);

  std::vector<size_t> col_pos = c.col_ptrs;
  col_pos.pop_back();

  tbb::combinable<std::vector<size_t>> comb_pos([col_pos] { return col_pos; });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, cols_sz, 64),
                    [&](const tbb::blocked_range<size_t> &range) { FillColumnValues(a, b, c, comb_pos, range); });

  return true;
}

}  // namespace kapanova_s_sparse_matrix_mult_ccs
