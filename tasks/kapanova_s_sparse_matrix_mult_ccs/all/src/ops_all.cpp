#include "kapanova_s_sparse_matrix_mult_ccs/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kapanova_s_sparse_matrix_mult_ccs/common/include/common.hpp"

namespace kapanova_s_sparse_matrix_mult_ccs {

KapanovaSSparseMatrixMultCCSALL::KapanovaSSparseMatrixMultCCSALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSSparseMatrixMultCCSALL::ValidationImpl() {
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

bool KapanovaSSparseMatrixMultCCSALL::PreProcessingImpl() {
  return true;
}
bool KapanovaSSparseMatrixMultCCSALL::PostProcessingImpl() {
  return true;
}

namespace {

void ProcessColumn(int j, const CCSMatrix &a, const CCSMatrix &b, std::vector<size_t> &out_rows,
                   std::vector<double> &out_vals) {
  std::vector<double> accum(a.rows, 0.0);
  std::vector<char> mask(a.rows, 0);
  std::vector<size_t> active;

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

  std::sort(active.begin(), active.end());

  for (size_t i : active) {
    if (accum[i] != 0.0) {
      out_rows.push_back(i);
      out_vals.push_back(accum[i]);
    }
  }
}

}  // namespace

bool KapanovaSSparseMatrixMultCCSALL::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  OutType &c = GetOutput();

  c.rows = a.rows;
  c.cols = b.cols;

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  size_t cols_per_proc = (c.cols + mpi_size - 1) / mpi_size;
  size_t start_col = mpi_rank * cols_per_proc;
  size_t end_col = std::min(start_col + cols_per_proc, static_cast<size_t>(c.cols));
  size_t local_cols = end_col - start_col;

  std::vector<std::vector<size_t>> temp_rows(local_cols);
  std::vector<std::vector<double>> temp_vals(local_cols);

  if (local_cols > 0) {
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t local_idx = 0; local_idx < local_cols; ++local_idx) {
      size_t j = start_col + local_idx;
      ProcessColumn(static_cast<int>(j), a, b, temp_rows[local_idx], temp_vals[local_idx]);
    }
  }

  std::vector<int> local_sizes(local_cols);
  for (size_t j = 0; j < local_cols; ++j) {
    local_sizes[j] = static_cast<int>(temp_rows[j].size());
  }

  std::vector<int> all_sizes;
  if (mpi_rank == 0) {
    all_sizes.resize(c.cols);
  }

  MPI_Gather(local_sizes.data(), static_cast<int>(local_cols), MPI_INT, all_sizes.data(), static_cast<int>(local_cols),
             MPI_INT, 0, MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    size_t offset = 0;
    c.col_ptrs.resize(c.cols + 1);
    for (int j = 0; j < static_cast<int>(c.cols); ++j) {
      c.col_ptrs[j] = offset;
      offset += all_sizes[j];
    }
    c.col_ptrs[c.cols] = offset;
    c.nnz = offset;
    c.values.resize(c.nnz);
    c.row_indices.resize(c.nnz);
  }

  std::vector<int> flat_rows;
  std::vector<double> flat_vals;
  for (size_t j = 0; j < local_cols; ++j) {
    for (size_t v : temp_rows[j]) {
      flat_rows.push_back(static_cast<int>(v));
    }
    flat_vals.insert(flat_vals.end(), temp_vals[j].begin(), temp_vals[j].end());
  }

  int local_nnz = static_cast<int>(flat_rows.size());
  std::vector<int> recv_counts(mpi_size, 0);
  std::vector<int> displs(mpi_size, 0);

  MPI_Gather(&local_nnz, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    displs[0] = 0;
    for (int i = 1; i < mpi_size; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
  }

  std::vector<int> all_rows;
  std::vector<double> all_vals;
  if (mpi_rank == 0) {
    int total = 0;
    for (int i = 0; i < mpi_size; ++i) {
      total += recv_counts[i];
    }
    all_rows.resize(total);
    all_vals.resize(total);
  }

  MPI_Gatherv(flat_rows.data(), local_nnz, MPI_INT, all_rows.data(), recv_counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);
  MPI_Gatherv(flat_vals.data(), local_nnz, MPI_DOUBLE, all_vals.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    std::vector<size_t> pos(c.cols, 0);
    size_t idx = 0;

    for (int p = 0; p < mpi_size; ++p) {
      size_t p_start = p * cols_per_proc;
      size_t p_end = std::min(p_start + cols_per_proc, static_cast<size_t>(c.cols));

      for (size_t local_j = 0; local_j < p_end - p_start; ++local_j) {
        size_t global_j = p_start + local_j;
        size_t offset = c.col_ptrs[global_j] + pos[global_j];
        int size = all_sizes[global_j];

        for (int k = 0; k < size; ++k) {
          c.row_indices[offset + k] = all_rows[idx + k];
          c.values[offset + k] = all_vals[idx + k];
        }
        idx += size;
        pos[global_j] += size;
      }
    }
  }

  if (mpi_rank == 0) {
    MPI_Bcast(c.col_ptrs.data(), c.cols + 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c.nnz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(c.row_indices.data(), static_cast<int>(c.nnz), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(c.values.data(), static_cast<int>(c.nnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    c.col_ptrs.resize(c.cols + 1);
    MPI_Bcast(c.col_ptrs.data(), c.cols + 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c.nnz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    c.row_indices.resize(c.nnz);
    c.values.resize(c.nnz);
    MPI_Bcast(c.row_indices.data(), static_cast<int>(c.nnz), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(c.values.data(), static_cast<int>(c.nnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace kapanova_s_sparse_matrix_mult_ccs
