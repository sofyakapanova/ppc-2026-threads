#pragma once

#include "task/include/task.hpp"
#include "zyazeva_s_matrix_mult_cannon_alg/common/include/common.hpp"

namespace zyazeva_s_matrix_mult_cannon_alg {

std::vector<double> CannonMatrixMultiplication(const std::vector<double> &a, const std::vector<double> &b, int n);

class ZyazevaSMatrixMultCannonAlgOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit ZyazevaSMatrixMultCannonAlgOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zyazeva_s_matrix_mult_cannon_alg
