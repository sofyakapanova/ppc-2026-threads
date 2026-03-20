#pragma once

#include "kapanova_s_sparse_matrix_mult_ccs_seq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kapanova_s_sparse_matrix_mult_ccs_seq {

class KapanovaSSparseMatrixMultCCSOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit KapanovaSSparseMatrixMultCCSOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kapanova_s_sparse_matrix_mult_ccs_seq