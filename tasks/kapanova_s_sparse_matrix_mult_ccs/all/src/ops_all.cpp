#include "kapanova_s_sparse_matrix_mult_ccs/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
  return (a.cols == b.rows && a.rows > 0 && a.cols > 0 && b.rows > 0 && b.cols > 0 &&
          a.col_ptrs.size() == static_cast<size_t>(a.cols + 1) && b.col_ptrs.size() == static_cast<size_t>(b.cols + 1));
}

bool KapanovaSSparseMatrixMultCCSALL::PreProcessingImpl() {
  return true;
}
bool KapanovaSSparseMatrixMultCCSALL::PostProcessingImpl() {
  return true;
}

namespace {

std::vector<size_t> ComputeBalancedRanges(int total_cols, int num_procs, const CCSMatrix &a, const CCSMatrix &b) {
  std::vector<size_t> ranges(num_procs + 1, 0);
  ranges[num_procs] = static_cast<size_t>(total_cols);

  if (total_cols == 0) {
    return ranges;
  }

  std::vector<int> col_cost(total_cols);
  int total_cost = 0;
#pragma omp parallel for reduction(+ : total_cost) schedule(guided)
  for (int j = 0; j < total_cols; ++j) {
    int cost = 0;
    for (size_t k = b.col_ptrs[j]; k < b.col_ptrs[j + 1]; ++k) {
      size_t row_b = b.row_indices[k];
      cost += static_cast<int>(a.col_ptrs[row_b + 1] - a.col_ptrs[row_b]);
    }
    col_cost[j] = cost;
    total_cost += cost;
  }

  int cost_per_proc = total_cost / num_procs;
  int current_col = 0;
  int accumulated_cost = 0;

  for (int p = 1; p < num_procs; ++p) {
    int target_cost = p * cost_per_proc;
    while (current_col < total_cols && accumulated_cost < target_cost) {
      accumulated_cost += col_cost[current_col];
      ++current_col;
    }
    ranges[p] = static_cast<size_t>(current_col);
  }

  for (int p = num_procs - 1; p > 0; --p) {
    if (ranges[p] == ranges[p - 1] && static_cast<int>(ranges[p]) < total_cols) {
      ranges[p]++;
    }
  }

  return ranges;
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

  std::vector<uint64_t> ranges(mpi_size + 1);
  if (mpi_rank == 0) {
    auto raw = ComputeBalancedRanges(c.cols, mpi_size, a, b);
    ranges.assign(raw.begin(), raw.end());
  }
  MPI_Bcast(ranges.data(), mpi_size + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  size_t start_col = ranges[mpi_rank];
  size_t end_col = ranges[mpi_rank + 1];
  size_t local_cols = end_col - start_col;

  std::vector<int> local_sizes(local_cols, 0);
  std::vector<std::vector<size_t>> temp_rows(local_cols);
  std::vector<std::vector<double>> temp_vals(local_cols);

  if (local_cols > 0) {
#pragma omp parallel
    {
      std::vector<double> accum(a.rows, 0.0);
      std::vector<char> mask(a.rows, 0);
      std::vector<size_t> active;
      active.reserve(a.rows / 10);

#pragma omp for schedule(guided, 32) nowait
      for (size_t local_idx = 0; local_idx < local_cols; ++local_idx) {
        size_t j = start_col + local_idx;

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
            temp_rows[local_idx].push_back(i);
            temp_vals[local_idx].push_back(accum[i]);
          }
          mask[i] = 0;
          accum[i] = 0.0;
        }
        local_sizes[local_idx] = static_cast<int>(temp_rows[local_idx].size());
        active.clear();
      }
    }
  }

  int local_nnz = 0;
  for (size_t j = 0; j < local_cols; ++j) {
    local_nnz += local_sizes[j];
  }

  std::vector<uint64_t> send_rows(local_nnz);
  std::vector<double> send_vals(local_nnz);

  size_t offset = 0;
  for (size_t j = 0; j < local_cols; ++j) {
    for (int k = 0; k < local_sizes[j]; ++k) {
      send_rows[offset + k] = static_cast<uint64_t>(temp_rows[j][k]);
      send_vals[offset + k] = temp_vals[j][k];
    }
    offset += local_sizes[j];
  }

  std::vector<int> recv_counts(mpi_size);
  std::vector<int> displs(mpi_size, 0);
  MPI_Gather(&local_nnz, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total_nnz = 0;
  if (mpi_rank == 0) {
    for (int i = 0; i < mpi_size; ++i) {
      displs[i] = total_nnz;
      total_nnz += recv_counts[i];
    }
    c.nnz = total_nnz;
    c.values.resize(c.nnz);
    c.row_indices.resize(c.nnz);
    c.col_ptrs.resize(c.cols + 1);
  }

  MPI_Gatherv(send_rows.data(), local_nnz, MPI_UINT64_T, c.row_indices.data(), recv_counts.data(), displs.data(),
              MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Gatherv(send_vals.data(), local_nnz, MPI_DOUBLE, c.values.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  std::vector<int> send_col_counts(local_cols);
  for (size_t j = 0; j < local_cols; ++j) {
    send_col_counts[j] = local_sizes[j];
  }

  int local_count = static_cast<int>(local_cols);
  std::vector<int> proc_counts(mpi_size);
  std::vector<int> col_displs(mpi_size, 0);
  MPI_Gather(&local_count, 1, MPI_INT, proc_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> all_col_sizes;
  if (mpi_rank == 0) {
    int total_cols_cnt = 0;
    for (int i = 0; i < mpi_size; ++i) {
      col_displs[i] = total_cols_cnt;
      total_cols_cnt += proc_counts[i];
    }
    all_col_sizes.resize(total_cols_cnt);
  }

  MPI_Gatherv(send_col_counts.data(), local_count, MPI_INT, all_col_sizes.data(), proc_counts.data(), col_displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    size_t off = 0;
    for (size_t j = 0; j < static_cast<size_t>(c.cols); ++j) {
      c.col_ptrs[j] = off;
      off += static_cast<size_t>(all_col_sizes[j]);
    }
    c.col_ptrs[c.cols] = off;
  }

  uint64_t nnz_tmp = c.nnz;
  uint64_t cols_p1 = static_cast<uint64_t>(c.cols + 1);
  MPI_Bcast(&nnz_tmp, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_p1, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (mpi_rank != 0) {
    c.col_ptrs.resize(cols_p1);
  }
  MPI_Bcast(c.col_ptrs.data(), cols_p1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (mpi_rank != 0) {
    c.cols = static_cast<int>(cols_p1 - 1);
    c.nnz = nnz_tmp;
    c.row_indices.resize(c.nnz);
    c.values.resize(c.nnz);
  }

  MPI_Bcast(c.row_indices.data(), static_cast<int>(c.nnz), MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(c.values.data(), static_cast<int>(c.nnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

}  // namespace kapanova_s_sparse_matrix_mult_ccs
