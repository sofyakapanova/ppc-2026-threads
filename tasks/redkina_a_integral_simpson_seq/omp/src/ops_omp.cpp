#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"

namespace redkina_a_integral_simpson_seq {

namespace {

inline double GetCoeff(int idx, int n) {
  if (idx == 0 || idx == n) {
    return 1.0;
  }
  return (idx % 2 == 1) ? 4.0 : 2.0;
}

constexpr int MAX_DIM = 10;

}  // namespace

RedkinaAIntegralSimpsonOMP::RedkinaAIntegralSimpsonOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RedkinaAIntegralSimpsonOMP::ValidationImpl() {
  const auto &in = GetInput();
  size_t dim = in.a.size();

  if (dim == 0 || in.b.size() != dim || in.n.size() != dim) {
    return false;
  }

  if (dim > MAX_DIM) {
    return false;
  }

  for (size_t i = 0; i < dim; ++i) {
    if (in.a[i] >= in.b[i]) {
      return false;
    }
    if (in.n[i] <= 0 || in.n[i] % 2 != 0) {
      return false;
    }
  }

  return static_cast<bool>(in.func);
}

bool RedkinaAIntegralSimpsonOMP::PreProcessingImpl() {
  const auto &in = GetInput();

  func_ = in.func;
  a_ = in.a;
  b_ = in.b;
  n_ = in.n;

  result_ = 0.0;

  return true;
}

bool RedkinaAIntegralSimpsonOMP::RunImpl() {
  const size_t dim = a_.size();

  const auto a_l = a_;
  const auto b_l = b_;
  const auto n_l = n_;
  const auto func_l = func_;

  double h[MAX_DIM];

  for (size_t i = 0; i < dim; ++i) {
    h[i] = (b_l[i] - a_l[i]) / static_cast<double>(n_l[i]);
  }

  double h_prod = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    h_prod *= h[i];
  }

  long long total_points = 1;
  for (size_t i = 0; i < dim; ++i) {
    total_points *= static_cast<long long>(n_l[i]) + 1;
  }

  double sum = 0.0;

#pragma omp parallel for default(none) reduction(+ : sum) shared(total_points, dim, a_l, n_l, h, func_l)
  for (long long linear_idx = 0; linear_idx < total_points; ++linear_idx) {
    int indices[MAX_DIM];
    double point[MAX_DIM];

    long long tmp = linear_idx;
    double weight_prod = 1.0;

    for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
      int size_d = n_l[d] + 1;
      indices[d] = static_cast<int>(tmp % size_d);
      tmp /= size_d;
    }

    for (size_t d = 0; d < dim; ++d) {
      point[d] = a_l[d] + indices[d] * h[d];
      weight_prod *= GetCoeff(indices[d], n_l[d]);
    }

    sum += weight_prod * func_l({point, point + dim});
  }

  double denominator = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    denominator *= 3.0;
  }

  result_ = (h_prod / denominator) * sum;

  return true;
}

bool RedkinaAIntegralSimpsonOMP::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace redkina_a_integral_simpson_seq
