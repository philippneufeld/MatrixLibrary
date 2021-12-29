// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Expressions_MatrixExpression_H_
#define ML_MATH_Expressions_MatrixExpression_H_

// Includes
#include "../MathPrerequisites.h"
#include "../Matrix.h"

#include "../../QTL/Type.h"
#include "../../QTL/EnableIf.h"

namespace ML
{
  template<typename ET>
  class TMLMatrixExpression : public TMLCRTP<ET> {};

  //
  // Traits
  //

  // Trait that checks if type is a matrix expression
  template<typename ExT>
  struct TMLIsMatrixExpression : TMLIsCRTP<ExT, TMLMatrixExpression> {};

  template<typename ExT>
  constexpr bool TMLIsMatrixExpression_v = TMLIsMatrixExpression<ExT>::value;

  // Get the inner CRTP type
  template<typename ExT, typename=void>
  struct TMLDecayMatrixExpressionType;
  template<typename ExT>
  struct TMLDecayMatrixExpressionType<ExT, TMLEnableIf_t<TMLIsMatrixExpression_v<ExT>>> 
    : TMLDecayCRTP<ExT> {};

  template<typename ExT>
  using TMLDecayMatrixExpressionType_t = typename TMLDecayMatrixExpressionType<ExT>::type;

  // Gets the result type of a matrix expression
  template<typename ExT, typename=void>
  struct TMLMatrixExpressionResultType;
  template<typename ExT>
  struct TMLMatrixExpressionResultType<ExT, TMLEnableIf_t<TMLIsMatrix_v<ExT>>>
    : TMLType<TMLDecayMatrixType_t<ExT>> {};
  template<typename ExT>
  struct TMLMatrixExpressionResultType<ExT, TMLEnableIf_t<TMLIsMatrixExpression_v<ExT>>>
    : TMLType<TMLDecayCRTPGeneral_t<typename TMLDecayMatrixExpressionType_t<ExT>::ResultType>> {};

  template<typename ExT>
  using TMLMatrixExpressionResultType_t = typename TMLMatrixExpressionResultType<ExT>::type;

}

#include "DMAssign.h"
#include "DMSetZero.h"
#include "DMSet1.h"
#include "DMDMAdd.h"
#include "DMDMMul.h"

#endif
