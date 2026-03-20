#include "kapanova_s_sparse_matrix_mult_ccs_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kapanova_s_sparse_matrix_mult_ccs_seq/common/include/common.hpp"

namespace kapanova_s_sparse_matrix_mult_ccs_seq {

KapanovaSSparseMatrixMultCCSOMP::KapanovaSSparseMatrixMultCCSOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSSparseMatrixMultCCSOMP::ValidationImpl() {
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
  if (a.col_ptrs[0] != 0 || b.col_ptrs[0] != 0) {
    return false;
  }
  if (a.col_ptrs[a.cols] != a.nnz) {
    return false;
  }
  if (b.col_ptrs[b.cols] != b.nnz) {
    return false;
  }
  if (a.values.size() != static_cast<size_t>(a.nnz) || a.row_indices.size() != static_cast<size_t>(a.nnz)) {
    return false;
  }
  if (b.values.size() != static_cast<size_t>(b.nnz) || b.row_indices.size() != static_cast<size_t>(b.nnz)) {
    return false;
  }

  return true;
}

bool KapanovaSSparseMatrixMultCCSOMP::PreProcessingImpl() {
  return true;
}

bool KapanovaSSparseMatrixMultCCSOMP::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  OutType &c = GetOutput();

  c.rows = a.rows;
  c.cols = b.cols;
  c.col_ptrs.assign(c.cols + 1, 0);
  c.nnz = 0;

  std::vector<std::vector<double>> col_vals(c.cols);
  std::vector<std::vector<size_t>> col_rows(c.cols);

#pragma omp parallel default(none) shared(a, b, col_vals, col_rows, c)
  {
    std::vector<double> local_accum(a.rows, 0.0);
    std::vector<bool> row_mask(a.rows, false);
    std::vector<size_t> active_rows;
    active_rows.reserve(a.rows);

#pragma omp for schedule(dynamic)
    for (int j = 0; j < static_cast<int>(c.cols); ++j) {
      active_rows.clear();

      for (size_t k = b.col_ptrs[j]; k < b.col_ptrs[j + 1]; ++k) {
        size_t row_b = b.row_indices[k];
        double val_b = b.values[k];

        for (size_t zc = a.col_ptrs[row_b]; zc < a.col_ptrs[row_b + 1]; ++zc) {
          size_t i = a.row_indices[zc];
          if (!row_mask[i]) {
            row_mask[i] = true;
            active_rows.push_back(i);
          }
          local_accum[i] += a.values[zc] * val_b;
        }
      }

      std::sort(active_rows.begin(), active_rows.end());

      for (size_t i : active_rows) {
        if (std::abs(local_accum[i]) > 1e-12) {
          col_vals[j].push_back(local_accum[i]);
          col_rows[j].push_back(i);
        }
        local_accum[i] = 0.0;
        row_mask[i] = false;
      }
    }
  }

  size_t offset = 0;
  for (size_t j = 0; j < c.cols; ++j) {
    c.col_ptrs[j] = offset;
    offset += col_vals[j].size();
  }
  c.col_ptrs[c.cols] = offset;
  c.nnz = offset;

  c.values.resize(c.nnz);
  c.row_indices.resize(c.nnz);

  for (size_t j = 0; j < c.cols; ++j) {
    size_t start = c.col_ptrs[j];
    for (size_t idx = 0; idx < col_vals[j].size(); ++idx) {
      c.values[start + idx] = col_vals[j][idx];
      c.row_indices[start + idx] = col_rows[j][idx];
    }
  }

  return true;
}

bool KapanovaSSparseMatrixMultCCSOMP::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_sparse_matrix_mult_ccs_seq
