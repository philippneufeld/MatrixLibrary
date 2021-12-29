// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Dense_DenseMatrix_H_
#define ML_MATH_Dense_DenseMatrix_H_

// Includes
#include "../Matrix.h"

namespace ML
{
  
  // Dense Matrix CRTP
  template<typename MT>
  class TMLDenseMatrix : public TMLMatrix<MT> {};

  //
  // Traits
  //

  // Is dense matrix?
  template<typename MT>
  struct TMLMatrixIsDense : TMLIsCRTP<MT, TMLDenseMatrix> {};

  template<typename MT>
  constexpr bool TMLMatrixIsDense_v = TMLMatrixIsDense<MT>::value;

}

#include "StaticMatrix.h"
#include "DynamicMatrix.h"

#endif