// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Dense_DenseMatrixHelper_H_
#define ML_MATH_Dense_DenseMatrixHelper_H_

// Includes
#include "DenseMatrix.h"
#include "../SIMD/SIMD.h"
#include "../Expressions/MatrixExpression.h"

namespace ML
{

  template<typename MT>
  class TMLDenseMatrixHelper : public TMLDenseMatrix<MT>
  {
  public:
    void SetZero() { (~(*this)).Assign(TMLDMSetZeroExpression<MT>((~(*this)).Rows(), (~(*this)).Cols())); }
    template<typename ET> void Set1(ET value) { (~(*this)).Assign(TMLDMSet1Expression<MT>(value, (~(*this)).Rows(), (~(*this)).Cols())); }
    const auto& Transpose() { return *reinterpret_cast<typename MT::TransposeType*>(&(~(*this))); }
  };

}

#endif
