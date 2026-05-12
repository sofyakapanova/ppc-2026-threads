#include "kapanova_s_sparse_matrix_mult_ccs/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
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

#ifdef _WIN32
using MpiSizeT = unsigned long long;
const auto kMpiSizeT = MPI_UNSIGNED_LONG_LONG;
#else
using MpiSizeT = size_t;
const auto kMpiSizeT = MPI_UNSIGNED_LONG;
#endif

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
      cost += static_cast<int>(a.col_ptrs[b.row_indices[k] + 1] - a.col_ptrs[b.row_indices[k]]);
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
    if (ranges[proc] == ranges[proc - 1] && std::cmp_less(ranges[proc], total_cols)) {
      ranges[proc]++;
    }
  }
  return ranges;
}

void ProcessColumn(size_t j, const CCSMatrix &a, const CCSMatrix &b, std::vector<size_t> &out_rows,
                   std::vector<double> &out_vals, double *accum, char *mask, size_t *active, int &active_count) {
  for (size_t k = b.col_ptrs[j]; k < b.col_ptrs[j + 1]; ++k) {
    size_t row_b = b.row_indices[k];
    double val_b = b.values[k];
    for (size_t zc = a.col_ptrs[row_b]; zc < a.col_ptrs[row_b + 1]; ++zc) {
      size_t i = a.row_indices[zc];
      double val_a = a.values[zc];
      if (mask[i] == 0) {
        mask[i] = 1;
        active[active_count++] = i;
        accum[i] = val_a * val_b;
      } else {
        accum[i] += val_a * val_b;
      }
    }
  }
  std::sort(active, active + active_count);
  for (int idx = 0; idx < active_count; ++idx) {
    size_t i = active[idx];
    if (accum[i] != 0.0) {
      out_rows.push_back(i);
      out_vals.push_back(accum[i]);
    }
    mask[i] = 0;
    accum[i] = 0.0;
  }
}

void ComputeLocalColumns(size_t start_col, size_t local_cols, const CCSMatrix &a, const CCSMatrix &b,
                         std::vector<int> &local_sizes, std::vector<std::vector<size_t>> &temp_rows,
                         std::vector<std::vector<double>> &temp_vals) {
#pragma omp parallel default(none) shared(a, b, start_col, local_cols, temp_rows, temp_vals, local_sizes)
  {
    std::vector<double> accum(a.rows, 0.0);
    std::vector<char> mask(a.rows, 0);
    std::vector<size_t> active(a.rows);
    int active_count = 0;
#pragma omp for schedule(guided, 32) nowait
    for (size_t local_idx = 0; local_idx < local_cols; ++local_idx) {
      active_count = 0;
      ProcessColumn(start_col + local_idx, a, b, temp_rows[local_idx], temp_vals[local_idx], accum.data(), mask.data(),
                    active.data(), active_count);
      local_sizes[local_idx] = static_cast<int>(temp_rows[local_idx].size());
    }
  }
}

int PackSendData(const std::vector<int> &local_sizes, size_t local_cols,
                 const std::vector<std::vector<size_t>> &temp_rows, const std::vector<std::vector<double>> &temp_vals,
                 std::vector<MpiSizeT> &send_rows, std::vector<double> &send_vals) {
  int total = 0;
  for (size_t j = 0; j < local_cols; ++j) {
    total += local_sizes[j];
  }
  send_rows.resize(total);
  send_vals.resize(total);
  size_t offset = 0;
  for (size_t j = 0; j < local_cols; ++j) {
    for (int k = 0; k < local_sizes[j]; ++k) {
      send_rows[offset + k] = static_cast<MpiSizeT>(temp_rows[j][k]);
      send_vals[offset + k] = temp_vals[j][k];
    }
    offset += static_cast<size_t>(local_sizes[j]);
  }
  return total;
}

void GatherRowValues(std::vector<size_t> &row_indices, std::vector<double> &values, size_t &nnz,
                     const std::vector<MpiSizeT> &send_rows, const std::vector<double> &send_vals, int local_nnz,
                     int mpi_rank, int mpi_size) {
  std::vector<int> recv_counts(mpi_size);
  MPI_Gather(&local_nnz, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> displs(mpi_size, 0);
  size_t total = 0;
  if (mpi_rank == 0) {
    for (int proc = 0; proc < mpi_size; ++proc) {
      displs[proc] = static_cast<int>(total);
      total += static_cast<size_t>(recv_counts[proc]);
    }
    nnz = total;
    values.resize(nnz);
  }
  std::vector<MpiSizeT> tmp_rows(mpi_rank == 0 ? total : 1);
  MPI_Gatherv(send_rows.data(), local_nnz, kMpiSizeT, tmp_rows.data(), recv_counts.data(), displs.data(), kMpiSizeT, 0,
              MPI_COMM_WORLD);
  MPI_Gatherv(send_vals.data(), local_nnz, MPI_DOUBLE, values.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    row_indices.resize(nnz);
    for (size_t i = 0; i < nnz; ++i) {
      row_indices[i] = static_cast<size_t>(tmp_rows[i]);
    }
  }
}

void GatherAndBroadcast(std::vector<size_t> &col_ptrs, std::vector<size_t> &row_indices, std::vector<double> &values,
                        size_t &nnz, int &cols, const std::vector<int> &local_sizes, int local_count, int mpi_rank,
                        int mpi_size) {
  std::vector<int> proc_counts(mpi_size);
  MPI_Gather(&local_count, 1, MPI_INT, proc_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> col_displs(mpi_size, 0);
  std::vector<int> all_col_sizes;
  if (mpi_rank == 0) {
    int total = 0;
    for (int proc = 0; proc < mpi_size; ++proc) {
      col_displs[proc] = total;
      total += proc_counts[proc];
    }
    all_col_sizes.resize(total);
    col_ptrs.resize(static_cast<size_t>(cols) + 1);
  }
  MPI_Gatherv(local_sizes.data(), local_count, MPI_INT, all_col_sizes.data(), proc_counts.data(), col_displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    size_t off = 0;
    for (int j = 0; j < cols; ++j) {
      col_ptrs[j] = off;
      off += static_cast<size_t>(all_col_sizes[j]);
    }
    col_ptrs[cols] = off;
  }
  auto nnz_bcast = static_cast<MpiSizeT>(nnz);
  auto cols_bcast = static_cast<MpiSizeT>(cols) + 1;
  MPI_Bcast(&nnz_bcast, 1, kMpiSizeT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_bcast, 1, kMpiSizeT, 0, MPI_COMM_WORLD);
  std::vector<MpiSizeT> cp_tmp(mpi_rank == 0 ? cols_bcast : 1);
  std::vector<MpiSizeT> ri_tmp(mpi_rank == 0 ? nnz_bcast : 1);
  if (mpi_rank == 0) {
    for (size_t i = 0; i < col_ptrs.size(); ++i) {
      cp_tmp[i] = static_cast<MpiSizeT>(col_ptrs[i]);
    }
    for (size_t i = 0; i < row_indices.size(); ++i) {
      ri_tmp[i] = static_cast<MpiSizeT>(row_indices[i]);
    }
  }
  MPI_Bcast(cp_tmp.data(), static_cast<int>(cols_bcast), kMpiSizeT, 0, MPI_COMM_WORLD);
  MPI_Bcast(ri_tmp.data(), static_cast<int>(nnz_bcast), kMpiSizeT, 0, MPI_COMM_WORLD);
  if (mpi_rank != 0) {
    cols = static_cast<int>(cols_bcast - 1);
    nnz = nnz_bcast;
    col_ptrs.assign(cp_tmp.begin(), cp_tmp.end());
    row_indices.assign(ri_tmp.begin(), ri_tmp.end());
    values.resize(nnz);
  }
  MPI_Bcast(values.data(), static_cast<int>(nnz_bcast), MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
  if (mpi_rank == 0) {
    ranges = ComputeBalancedRanges(static_cast<int>(c.cols), mpi_size, a, b);
  }
  MPI_Bcast(ranges.data(), mpi_size + 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  size_t start_col = ranges[mpi_rank];
  size_t local_cols = ranges[mpi_rank + 1] - start_col;

  std::vector<int> local_sizes(local_cols, 0);
  std::vector<std::vector<size_t>> temp_rows(local_cols);
  std::vector<std::vector<double>> temp_vals(local_cols);

  if (local_cols > 0) {
    ComputeLocalColumns(start_col, local_cols, a, b, local_sizes, temp_rows, temp_vals);
  }

  std::vector<MpiSizeT> send_rows;
  std::vector<double> send_vals;
  int local_nnz = PackSendData(local_sizes, local_cols, temp_rows, temp_vals, send_rows, send_vals);

  size_t nnz_tmp = 0;
  GatherRowValues(c.row_indices, c.values, nnz_tmp, send_rows, send_vals, local_nnz, mpi_rank, mpi_size);

  int cols_tmp = static_cast<int>(c.cols);
  GatherAndBroadcast(c.col_ptrs, c.row_indices, c.values, nnz_tmp, cols_tmp, local_sizes, static_cast<int>(local_cols),
                     mpi_rank, mpi_size);

  c.nnz = nnz_tmp;
  c.cols = cols_tmp;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

}  // namespace kapanova_s_sparse_matrix_mult_ccs
