#include "buzulukski_d_gaus_gorizontal/omp/include/ops_omp.hpp"

#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "buzulukski_d_gaus_gorizontal/common/include/common.hpp"

namespace buzulukski_d_gaus_gorizontal {

namespace {
constexpr int kChannels = 3;
constexpr int kKernelSize = 3;
constexpr int kKernelSum = 16;

using KernelRow = std::array<int, kKernelSize>;
constexpr std::array<KernelRow, kKernelSize> kKernel = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
}  // namespace

BuzulukskiDGausGorizontalOMP::BuzulukskiDGausGorizontalOMP(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BuzulukskiDGausGorizontalOMP::ValidationImpl() {
  return GetInput() >= kKernelSize;
}

bool BuzulukskiDGausGorizontalOMP::PreProcessingImpl() {
  width_ = GetInput();
  height_ = GetInput();

  if (width_ < kKernelSize) {
    return false;
  }

  const auto total_size = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * kChannels;
  input_image_.assign(total_size, static_cast<uint8_t>(100));
  output_image_.assign(total_size, 0);
  return true;
}

void BuzulukskiDGausGorizontalOMP::ApplyGaussianToPixel(int py, int px) {
  const auto u_width = static_cast<std::size_t>(width_);
  const auto u_height = static_cast<std::size_t>(height_);
  const auto u_channels = static_cast<std::size_t>(kChannels);

  for (int ch = 0; ch < kChannels; ++ch) {
    int sum = 0;
    for (int ky = -1; ky <= 1; ++ky) {
      for (int kx = -1; kx <= 1; ++kx) {
        const int ny = std::clamp(py + ky, 0, static_cast<int>(u_height) - 1);
        const int nx = std::clamp(px + kx, 0, static_cast<int>(u_width) - 1);

        const auto idx = (((static_cast<std::size_t>(ny) * u_width) + static_cast<std::size_t>(nx)) * u_channels) +
                         static_cast<std::size_t>(ch);

        const auto row_idx = static_cast<std::size_t>(ky) + 1;
        const auto col_idx = static_cast<std::size_t>(kx) + 1;

        sum += static_cast<int>(input_image_.at(idx)) * kKernel.at(row_idx).at(col_idx);
      }
    }
    const auto out_idx = (((static_cast<std::size_t>(py) * u_width) + static_cast<std::size_t>(px)) * u_channels) +
                         static_cast<std::size_t>(ch);
    output_image_.at(out_idx) = static_cast<uint8_t>(sum / kKernelSum);
  }
}

bool BuzulukskiDGausGorizontalOMP::RunImpl() {
#pragma omp parallel for
  for (int py = 0; py < height_; ++py) {
    for (int px = 0; px < width_; ++px) {
      ApplyGaussianToPixel(py, px);
    }
  }
  return true;
}

bool BuzulukskiDGausGorizontalOMP::PostProcessingImpl() {
  if (output_image_.empty()) {
    return false;
  }

  int64_t total_sum = 0;
  for (const auto &val : output_image_) {
    total_sum += static_cast<int64_t>(val);
  }
  GetOutput() = static_cast<int>(total_sum / static_cast<int64_t>(output_image_.size()));
  return true;
}

}  // namespace buzulukski_d_gaus_gorizontal