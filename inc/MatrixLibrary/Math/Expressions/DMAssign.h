// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Expressions_DMAssign_H_
#define ML_MATH_Expressions_DMAssign_H_

// Includes
#include "MatrixExpression.h"
#include "../Dense/DenseMatrix.h"

#include "../../QTL/EnableIf.h"

namespace ML
{
  
  template<typename M1>
  class TMLDMAssignExpression : public TMLMatrixExpression<TMLDMAssignExpression<M1>>
  {
  public:
    using MyT = TMLDMAssignExpression<M1>;
    using LOpType = TMLDecayMatrixType_t<M1>;
    using ResultType = TMLDecayMatrixType_t<M1>;
    
    using ElementType = TMLMatrixElementType_t<M1>;
    using SIMDType = TMLMatrixSIMDType_t<M1>;

    constexpr TMLDMAssignExpression(const LOpType& lhs) : m_lhs(lhs) {}

    constexpr std::size_t Rows() const noexcept { return (~m_lhs).Rows(); }
    constexpr std::size_t Cols() const noexcept { return (~m_lhs).Cols(); }

    ElementType operator()(std::size_t i, std::size_t j) const noexcept;

    template<typename MT>
    void AssignTo(TMLDenseMatrix<MT>& res) const;

  private:
    template<typename MT>
    constexpr static bool IsVectorizable_v = 
      TMLMatrixIsDense_v<MT> &&
      TMLMatrixIsDense_v<LOpType> &&
      TMLMatrixIsVectorized_v<MT> &&
      TMLMatrixIsVectorized_v<LOpType> && 
      TMLMatrixIsSameSIMDType_v<MT, LOpType> && 
      TMLMatrixIsRowMajor_v<MT> == TMLMatrixIsRowMajor_v<LOpType>;

    // Selects the right kernel
    template<typename MT> TMLEnableIf_t<!IsVectorizable_v<MT>, void> 
      ExecuteKernel(TMLDenseMatrix<MT>& res, const LOpType& lhs) const { DefaultKernel(res, lhs); }
    template<typename MT> TMLEnableIf_t<IsVectorizable_v<MT>, void>
      ExecuteKernel(TMLDenseMatrix<MT>& res, const LOpType& lhs) const { VectorizedKernel(res, lhs); }

    template<typename MT>
    void DefaultKernel(TMLDenseMatrix<MT>& res, const LOpType& lhs) const;
    template<typename MT, typename=TMLEnableIf_t<IsVectorizable_v<MT>>>
    void VectorizedKernel(TMLDenseMatrix<MT>& res, const LOpType& lhs) const;
    
    const LOpType& m_lhs;
  };


  template<typename M1> typename TMLDMAssignExpression<M1>::ElementType 
    TMLDMAssignExpression<M1>::operator()(std::size_t i, std::size_t j) const noexcept
  {
    assert(i < Rows());
    assert(j < Cols());
    return m_lhs(i, j);
  }

  template<typename M1>
  template<typename MT>
    void TMLDMAssignExpression<M1>::AssignTo(TMLDenseMatrix<MT>& res) const
  {
    assert((~res).Rows() == (~m_lhs).Rows());
    assert((~res).Cols() == (~m_lhs).Cols());
    
    // self assignment detection
    const void* resAddr = static_cast<const void*>(&(~res));
    const void* lhsAddr = static_cast<const void*>(&(~m_lhs));
    if (resAddr == lhsAddr)
      return;
    
    ExecuteKernel(res, m_lhs);
  }

  template<typename M1>
  template<typename MT>
  void TMLDMAssignExpression<M1>::DefaultKernel(
    TMLDenseMatrix<MT>& res, const LOpType& lhs) const
  {
    for (size_t i = 0; i < (~lhs).Rows(); i++)
    {
      for (size_t j = 0; j < (~lhs).Cols(); j++)
      {
        (~res)(i, j) = (~lhs)(i, j);
      }
    }
  }

  template<typename M1>
  template<typename MT, typename>
  void TMLDMAssignExpression<M1>::VectorizedKernel(
    TMLDenseMatrix<MT>& res, const LOpType& lhs) const
  {
    for (size_t i = 0; i < (~res).Rows(); i++)
    {
      for (size_t j = 0; j < (~res).Cols(); j += TMLSIMDSize_v<SIMDType>)
      {
        (~res).Store((~lhs).Load(i, j), i, j);
      }
    }
  }
  

}

#endif
