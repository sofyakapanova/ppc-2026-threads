#pragma once

#include <cstddef>
#include <vector>

#include "shkrebko_m_calc_of_integral_rect/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkrebko_m_calc_of_integral_rect {

class ShkrebkoMCalcOfIntegralRectTBB : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kTBB;
  }

  explicit ShkrebkoMCalcOfIntegralRectTBB(const InType &in);

 protected:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double ComputeBlockSum(std::size_t start_idx, std::size_t end_idx, const std::vector<double>& h);

  InType local_input_;
  double res_ = 0.0;
};

}  // namespace shkrebko_m_calc_of_integral_rect