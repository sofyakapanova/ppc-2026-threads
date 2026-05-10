#include "kapanova_s_sparse_matrix_mult_ccs/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
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
#pragma omp parallel for reduction(+ : total_cost) schedule(guided) default(none) shared(a, b, col_cost, total_cols)
  for (int col = 0; col < total_cols; ++col) {
    int cost = 0;
    for (size_t k = b.col_ptrs[col]; k < b.col_ptrs[col + 1]; ++k) {
      size_t row_b = b.row_indices[k];
      cost += static_cast<int>(a.col_ptrs[row_b + 1] - a.col_ptrs[row_b]);
    }
    col_cost[col] = cost;
    total_cost += cost;
  }

  int cost_per_proc = total_cost / num_procs;
  int current_col = 0;
  int accumulated_cost = 0;

  for (int proc = 1; proc < num_procs; ++proc) {
    int target_cost = proc * cost_per_proc;
    while (current_col < total_cols && accumulated_cost < target_cost) {
      accumulated_cost += col_cost[current_col];
      ++current_col;
    }
    ranges[proc] = static_cast<size_t>(current_col);
  }

  for (int proc = num_procs - 1; proc > 0; --proc) {
    if (ranges[proc] == ranges[proc - 1] && static_cast<int>(ranges[proc]) < total_cols) {
      ranges[proc]++;
    }
  }

  return ranges;
}

void ProcessSingleColumn(size_t j, const CCSMatrix &a, const CCSMatrix &b, std::vector<size_t> &out_rows,
                         std::vector<double> &out_vals, std::vector<double> &accum, std::vector<char> &mask,
                         std::vector<size_t> &active) {
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
    mask[i] = 0;
    accum[i] = 0.0;
  }
  active.clear();
}

void ComputeLocalColumns(size_t start_col, size_t local_cols, const CCSMatrix &a, const CCSMatrix &b,
                         std::vector<int> &local_sizes, std::vector<std::vector<size_t>> &temp_rows,
                         std::vector<std::vector<double>> &temp_vals) {
#pragma omp parallel default(none) shared(a, b, start_col, local_cols, temp_rows, temp_vals, local_sizes)
  {
    std::vector<double> accum(a.rows, 0.0);
    std::vector<char> mask(a.rows, 0);
    std::vector<size_t> active;
    active.reserve(a.rows / 10);

#pragma omp for schedule(guided, 32) nowait
    for (size_t local_idx = 0; local_idx < local_cols; ++local_idx) {
      size_t j = start_col + local_idx;
      ProcessSingleColumn(j, a, b, temp_rows[local_idx], temp_vals[local_idx], accum, mask, active);
      local_sizes[local_idx] = static_cast<int>(temp_rows[local_idx].size());
    }
  }
}

void DistributeRanges(std::vector<size_t> &ranges, int total_cols, int mpi_size, int mpi_rank, const CCSMatrix &a,
                      const CCSMatrix &b) {
  if (mpi_rank == 0) {
    ranges = ComputeBalancedRanges(total_cols, mpi_size, a, b);
  }
  std::vector<uint64_t> ranges_u64(ranges.begin(), ranges.end());
  MPI_Bcast(ranges_u64.data(), mpi_size + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  ranges.assign(ranges_u64.begin(), ranges_u64.end());
}

void PackSendData(const std::vector<int> &local_sizes, size_t local_cols,
                  const std::vector<std::vector<size_t>> &temp_rows, const std::vector<std::vector<double>> &temp_vals,
                  std::vector<size_t> &send_rows, std::vector<double> &send_vals) {
  size_t offset = 0;
  for (size_t j = 0; j < local_cols; ++j) {
    for (int k = 0; k < local_sizes[j]; ++k) {
      send_rows[offset + k] = temp_rows[j][k];
      send_vals[offset + k] = temp_vals[j][k];
    }
    offset += static_cast<size_t>(local_sizes[j]);
  }
}

void GatherRowValues(std::vector<size_t> &row_indices, std::vector<double> &values, size_t &nnz,
                     const std::vector<size_t> &send_rows, const std::vector<double> &send_vals, int local_nnz,
                     int mpi_rank, int mpi_size) {
  std::vector<int> recv_counts(mpi_size);
  MPI_Gather(&local_nnz, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs(mpi_size, 0);
  int total_nnz = 0;
  if (mpi_rank == 0) {
    for (int proc = 0; proc < mpi_size; ++proc) {
      displs[proc] = total_nnz;
      total_nnz += recv_counts[proc];
    }
    nnz = static_cast<size_t>(total_nnz);
    row_indices.resize(nnz);
    values.resize(nnz);
  }

  std::vector<uint64_t> send_rows_u64(send_rows.begin(), send_rows.end());
  MPI_Gatherv(send_rows_u64.data(), local_nnz, MPI_UINT64_T, row_indices.data(), recv_counts.data(), displs.data(),
              MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Gatherv(send_vals.data(), local_nnz, MPI_DOUBLE, values.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
}

void GatherColPtrs(std::vector<size_t> &col_ptrs, int total_cols, const std::vector<int> &local_sizes, int local_count,
                   int mpi_rank, int mpi_size) {
  std::vector<int> proc_counts(mpi_size);
  MPI_Gather(&local_count, 1, MPI_INT, proc_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> col_displs(mpi_size, 0);
  std::vector<int> all_col_sizes;
  if (mpi_rank == 0) {
    int total_cols_cnt = 0;
    for (int proc = 0; proc < mpi_size; ++proc) {
      col_displs[proc] = total_cols_cnt;
      total_cols_cnt += proc_counts[proc];
    }
    all_col_sizes.resize(total_cols_cnt);
    col_ptrs.resize(total_cols + 1);
  }

  MPI_Gatherv(local_sizes.data(), local_count, MPI_INT, all_col_sizes.data(), proc_counts.data(), col_displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    size_t off = 0;
    for (int j = 0; j < total_cols; ++j) {
      col_ptrs[j] = off;
      off += static_cast<size_t>(all_col_sizes[j]);
    }
    col_ptrs[total_cols] = off;
  }
}

void BroadcastResult(int total_cols, size_t nnz, std::vector<size_t> &col_ptrs, std::vector<size_t> &row_indices,
                     std::vector<double> &values, int root) {
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  auto nnz_u64 = static_cast<uint64_t>(nnz);
  auto cols_p1 = static_cast<uint64_t>(total_cols + 1);
  MPI_Bcast(&nnz_u64, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
  MPI_Bcast(&cols_p1, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);

  std::vector<uint64_t> cp_tmp(cols_p1);
  std::vector<uint64_t> ri_tmp(nnz_u64);

  if (mpi_rank == root) {
    for (size_t i = 0; i < col_ptrs.size(); ++i) {
      cp_tmp[i] = static_cast<uint64_t>(col_ptrs[i]);
    }
    for (size_t i = 0; i < row_indices.size(); ++i) {
      ri_tmp[i] = static_cast<uint64_t>(row_indices[i]);
    }
  }

  MPI_Bcast(cp_tmp.data(), static_cast<int>(cols_p1), MPI_UINT64_T, root, MPI_COMM_WORLD);
  MPI_Bcast(ri_tmp.data(), static_cast<int>(nnz_u64), MPI_UINT64_T, root, MPI_COMM_WORLD);

  if (mpi_rank != root) {
    col_ptrs.assign(cp_tmp.begin(), cp_tmp.end());
    row_indices.assign(ri_tmp.begin(), ri_tmp.end());
    values.resize(nnz_u64);
  }

  MPI_Bcast(values.data(), static_cast<int>(nnz_u64), MPI_DOUBLE, root, MPI_COMM_WORLD);
}

}  // namespace

bool KapanovaSSparseMatrixMultCCSALL::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  auto &c = GetOutput();

  c.rows = a.rows;
  c.cols = b.cols;

  int mpi_rank = 0;
  int mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::vector<size_t> ranges(mpi_size + 1);
  DistributeRanges(ranges, static_cast<int>(c.cols), mpi_size, mpi_rank, a, b);

  auto start_col = ranges[mpi_rank];
  auto end_col = ranges[mpi_rank + 1];
  auto local_cols = end_col - start_col;

  std::vector<int> local_sizes(local_cols, 0);
  std::vector<std::vector<size_t>> temp_rows(local_cols);
  std::vector<std::vector<double>> temp_vals(local_cols);

  if (local_cols > 0) {
    ComputeLocalColumns(start_col, local_cols, a, b, local_sizes, temp_rows, temp_vals);
  }

  int local_nnz = std::accumulate(local_sizes.begin(), local_sizes.end(), 0);

  std::vector<size_t> send_rows(local_nnz);
  std::vector<double> send_vals(local_nnz);
  PackSendData(local_sizes, local_cols, temp_rows, temp_vals, send_rows, send_vals);

  GatherRowValues(c.row_indices, c.values, c.nnz, send_rows, send_vals, local_nnz, mpi_rank, mpi_size);

  GatherColPtrs(c.col_ptrs, c.cols, local_sizes, static_cast<int>(local_cols), mpi_rank, mpi_size);

  BroadcastResult(c.cols, c.nnz, c.col_ptrs, c.row_indices, c.values, 0);

  if (mpi_rank != 0) {
    c.cols = static_cast<int>(c.col_ptrs.size() - 1);
    c.nnz = c.row_indices.size();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

}  // namespace kapanova_s_sparse_matrix_mult_ccs
