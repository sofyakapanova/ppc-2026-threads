#include "timur_a_cannon/stl/include/ops_stl.hpp"

#include <cstddef>
#include <future>
#include <thread>
#include <utility>
#include <vector>

#include "timur_a_cannon/common/include/common.hpp"

namespace timur_a_cannon {

namespace {

using Matrix = std::vector<std::vector<double>>;
using BlockGrid = std::vector<std::vector<Matrix>>;

void MultiplyBlocks(const Matrix &a, const Matrix &b, Matrix &c, int b_size) {
  for (int row = 0; row < b_size; ++row) {
    for (int k = 0; k < b_size; ++k) {
      double temp = a[row][k];
      for (int col = 0; col < b_size; ++col) {
        c[row][col] += temp * b[k][col];
      }
    }
  }
}

void RotateAll(BlockGrid &bl_a, BlockGrid &bl_b, int grid_sz) {
  for (int i = 0; i < grid_sz; ++i) {
    Matrix first = std::move(bl_a[i][0]);
    for (int j = 0; j < grid_sz - 1; ++j) {
      bl_a[i][j] = std::move(bl_a[i][j + 1]);
    }
    bl_a[i][grid_sz - 1] = std::move(first);
  }

  for (int j = 0; j < grid_sz; ++j) {
    Matrix first = std::move(bl_b[0][j]);
    for (int i = 0; i < grid_sz - 1; ++i) {
      bl_b[i][j] = std::move(bl_b[i + 1][j]);
    }
    bl_b[grid_sz - 1][j] = std::move(first);
  }
}

}  // namespace

TimurACannonMatrixMultiplicationSTL::TimurACannonMatrixMultiplicationSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TimurACannonMatrixMultiplicationSTL::ValidationImpl() {
  const auto &input = GetInput();
  int b_size = std::get<0>(input);
  const auto &mat_a = std::get<1>(input);
  const auto &mat_b = std::get<2>(input);

  if (b_size <= 0 || mat_a.empty() || mat_b.empty()) {
    return false;
  }

  size_t n = mat_a.size();
  return mat_a[0].size() == n && mat_b.size() == n && (n % static_cast<size_t>(b_size) == 0);
}

bool TimurACannonMatrixMultiplicationSTL::PreProcessingImpl() {
  return true;
}

bool TimurACannonMatrixMultiplicationSTL::RunImpl() {
  const auto &input = GetInput();
  int b_size = std::get<0>(input);
  const auto &matrix_a = std::get<1>(input);
  const auto &matrix_b = std::get<2>(input);
  int n = static_cast<int>(matrix_a.size());
  int grid_sz = n / b_size;

  BlockGrid bl_a(grid_sz, std::vector<Matrix>(grid_sz, Matrix(b_size, std::vector<double>(b_size))));
  BlockGrid bl_b(grid_sz, std::vector<Matrix>(grid_sz, Matrix(b_size, std::vector<double>(b_size))));
  BlockGrid bl_c(grid_sz, std::vector<Matrix>(grid_sz, Matrix(b_size, std::vector<double>(b_size, 0.0))));

  for (int i = 0; i < grid_sz; ++i) {
    for (int j = 0; j < grid_sz; ++j) {
      int s = (i + j) % grid_sz;
      for (int row = 0; row < b_size; ++row) {
        for (int col = 0; col < b_size; ++col) {
          bl_a[i][j][row][col] = matrix_a[(i * b_size) + row][(s * b_size) + col];
          bl_b[i][j][row][col] = matrix_b[(s * b_size) + row][(j * b_size) + col];
        }
      }
    }
  }

  for (int step = 0; step < grid_sz; ++step) {
    std::vector<std::future<void>> futures;
    futures.reserve(grid_sz * grid_sz);

    for (int i = 0; i < grid_sz; ++i) {
      for (int j = 0; j < grid_sz; ++j) {
        futures.push_back(std::async(std::launch::async,
                                     [&, i, j]() { MultiplyBlocks(bl_a[i][j], bl_b[i][j], bl_c[i][j], b_size); }));
      }
    }

    for (auto &f : futures) {
      f.wait();
    }

    if (grid_sz > 1 && step < grid_sz - 1) {
      RotateAll(bl_a, bl_b, grid_sz);
    }
  }

  Matrix result(n, std::vector<double>(n));
  for (int i = 0; i < grid_sz; ++i) {
    for (int j = 0; j < grid_sz; ++j) {
      for (int row = 0; row < b_size; ++row) {
        for (int col = 0; col < b_size; ++col) {
          result[(i * b_size) + row][(j * b_size) + col] = bl_c[i][j][row][col];
        }
      }
    }
  }

  GetOutput() = std::move(result);
  return true;
}

bool TimurACannonMatrixMultiplicationSTL::PostProcessingImpl() {
  return true;
}

}  // namespace timur_a_cannon
