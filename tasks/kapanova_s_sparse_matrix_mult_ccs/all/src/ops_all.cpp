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

using MpiU64 = std::uint64_t;
MPI_Datatype kMpiU64 = MPI_UINT64_T;

std::vector<MpiU64> ComputeBalancedRanges(int total_cols, int num_procs, const CCSMatrix &a, const CCSMatrix &b) {
  std::vector<MpiU64> ranges(static_cast<size_t>(num_procs) + 1, 0);
  ranges[num_procs] = static_cast<MpiU64>(total_cols);
  if (total_cols == 0) {
    return ranges;
  }

  std::vector<int> cost(static_cast<size_t>(total_cols), 0);
  int total_cost = 0;
#pragma omp parallel for reduction(+ : total_cost) schedule(guided)
  for (int j = 0; j < total_cols; ++j) {
    int c = 0;
    for (size_t k = b.col_ptrs[j]; k < b.col_ptrs[j + 1]; ++k) {
      c += static_cast<int>(a.col_ptrs[b.row_indices[k] + 1] - a.col_ptrs[b.row_indices[k]]);
    }
    cost[static_cast<size_t>(j)] = c;
    total_cost += c;
  }
  if (total_cost == 0) {
    return ranges;
  }
  int per = total_cost / num_procs;
  int cur = 0;
  int acc = 0;
  for (int p = 1; p < num_procs; ++p) {
    int target = p * per;
    while (cur < total_cols && acc < target) {
      acc += cost[static_cast<size_t>(cur)];
      ++cur;
    }
    ranges[static_cast<size_t>(p)] = static_cast<MpiU64>(cur);
  }
  return ranges;
}

}  // namespace

bool KapanovaSSparseMatrixMultCCSALL::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  auto &c = GetOutput();
  c.rows = a.rows;
  c.cols = b.cols;

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<MpiU64> ranges(size + 1, 0);
  if (rank == 0) {
    ranges = ComputeBalancedRanges(c.cols, size, a, b);
  }
  MPI_Bcast(ranges.data(), size + 1, kMpiU64, 0, MPI_COMM_WORLD);

  const size_t start = static_cast<size_t>(ranges[rank]);
  const size_t local_cols = static_cast<size_t>(ranges[rank + 1]) - start;

  std::vector<MpiU64> send_rows;
  std::vector<MpiU64> send_cols;
  std::vector<double> send_vals;

  if (local_cols > 0) {
#pragma omp parallel
    {
      std::vector<double> accum(a.rows, 0.0);
      std::vector<char> used(a.rows, 0);
      std::vector<size_t> active(static_cast<size_t>(a.rows));
      std::vector<MpiU64> thr_rows;
      std::vector<MpiU64> thr_cols;
      std::vector<double> thr_vals;

#pragma omp for schedule(guided) nowait
      for (size_t j = 0; j < local_cols; ++j) {
        int ac = 0;
        const size_t gcol = start + j;
        for (size_t k = b.col_ptrs[gcol]; k < b.col_ptrs[gcol + 1]; ++k) {
          size_t row_b = b.row_indices[k];
          double vb = b.values[k];
          for (size_t z = a.col_ptrs[row_b]; z < a.col_ptrs[row_b + 1]; ++z) {
            size_t i = a.row_indices[z];
            double va = a.values[z];
            if (!used[i]) {
              used[i] = 1;
              active[ac++] = i;
              accum[i] = va * vb;
            } else {
              accum[i] += va * vb;
            }
          }
        }
        for (int t = 0; t < ac; ++t) {
          size_t i = active[static_cast<size_t>(t)];
          if (accum[i] != 0.0) {
            thr_rows.push_back(static_cast<MpiU64>(i));
            thr_cols.push_back(static_cast<MpiU64>(gcol));
            thr_vals.push_back(accum[i]);
          }
          used[i] = 0;
          accum[i] = 0.0;
        }
      }
#pragma omp critical
      {
        send_rows.insert(send_rows.end(), thr_rows.begin(), thr_rows.end());
        send_cols.insert(send_cols.end(), thr_cols.begin(), thr_cols.end());
        send_vals.insert(send_vals.end(), thr_vals.begin(), thr_vals.end());
      }
    }
  }

  int local_nnz = static_cast<int>(send_rows.size());

  std::vector<int> counts(size, 0);
  MPI_Gather(&local_nnz, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs(size, 0);
  int total = 0;
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      displs[i] = total;
      total += counts[i];
    }
  }

  std::vector<MpiU64> recv_rows(rank == 0 ? static_cast<size_t>(total) : 0);
  std::vector<MpiU64> recv_cols(rank == 0 ? static_cast<size_t>(total) : 0);
  std::vector<double> recv_vals(rank == 0 ? static_cast<size_t>(total) : 0);

  const MpiU64 *rp = send_rows.empty() ? nullptr : send_rows.data();
  const MpiU64 *cp = send_cols.empty() ? nullptr : send_cols.data();
  const double *vp = send_vals.empty() ? nullptr : send_vals.data();

  MPI_Gatherv(rp, local_nnz, kMpiU64, recv_rows.data(), counts.data(), displs.data(), kMpiU64, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cp, local_nnz, kMpiU64, recv_cols.data(), counts.data(), displs.data(), kMpiU64, 0, MPI_COMM_WORLD);
  MPI_Gatherv(vp, local_nnz, MPI_DOUBLE, recv_vals.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    c.nnz = static_cast<size_t>(total);
    c.col_ptrs.assign(static_cast<size_t>(c.cols) + 1, 0);
    c.row_indices.resize(static_cast<size_t>(total));
    c.values.resize(static_cast<size_t>(total));

    if (total > 0) {
      std::vector<size_t> idx(static_cast<size_t>(total));
      std::iota(idx.begin(), idx.end(), static_cast<size_t>(0));
      std::sort(idx.begin(), idx.end(), [&](size_t a_idx, size_t b_idx) {
        if (recv_cols[a_idx] != recv_cols[b_idx]) {
          return recv_cols[a_idx] < recv_cols[b_idx];
        }
        return recv_rows[a_idx] < recv_rows[b_idx];
      });
      for (size_t i = 0; i < static_cast<size_t>(total); ++i) {
        size_t src = idx[i];
        c.row_indices[i] = static_cast<size_t>(recv_rows[src]);
        c.values[i] = recv_vals[src];
        c.col_ptrs[recv_cols[src] + 1]++;
      }
      for (size_t j = 0; j < static_cast<size_t>(c.cols); ++j) {
        c.col_ptrs[j + 1] += c.col_ptrs[j];
      }
    }
  }

  MpiU64 nnz_b = static_cast<MpiU64>(c.nnz);
  MpiU64 cols_b = static_cast<MpiU64>(c.cols);
  MPI_Bcast(&nnz_b, 1, kMpiU64, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_b, 1, kMpiU64, 0, MPI_COMM_WORLD);

  int nnz_i = static_cast<int>(nnz_b);
  int cols_i = static_cast<int>(cols_b);

  if (rank != 0) {
    c.nnz = static_cast<size_t>(nnz_b);
    c.cols = cols_i;
    c.col_ptrs.resize(static_cast<size_t>(cols_i) + 1);
    c.row_indices.resize(static_cast<size_t>(nnz_i));
    c.values.resize(static_cast<size_t>(nnz_i));
  }

  MPI_Bcast(c.col_ptrs.data(), cols_i + 1, kMpiU64, 0, MPI_COMM_WORLD);
  MPI_Bcast(c.row_indices.data(), nnz_i, kMpiU64, 0, MPI_COMM_WORLD);
  MPI_Bcast(c.values.data(), nnz_i, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

}  // namespace kapanova_s_sparse_matrix_mult_ccs
