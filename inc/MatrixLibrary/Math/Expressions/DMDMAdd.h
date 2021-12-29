// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Expressions_DMDMAdd_H_
#define ML_MATH_Expressions_DMDMAdd_H_

// Includes
#include <type_traits>
#include <cassert>

#include "MatrixExpression.h"
#include "../Matrix.h"

#include "../../QTL/EnableIf.h"

namespace ML
{

  template<typename M1, typename M2>
  class TMLDMDMAddExpression : public TMLMatrixExpression<TMLDMDMAddExpression<M1, M2>>
  {
    template<typename ML, typename MR, typename>
    friend auto operator+(const ML& lhs, const MR& rhs);

  public:
    using MyT = TMLDMDMAddExpression<M1, M2>;
    using LOpType = TMLDecayCRTP_t<M1>;
    using ROpType = TMLDecayCRTP_t<M2>;
    using LOpResType = TMLMatrixExpressionResultType_t<LOpType>;
    using ROpResType = TMLMatrixExpressionResultType_t<ROpType>;
    using ResultType = TMLMatrixAddResult_t<LOpResType, ROpResType>;

    using ElementType = TMLMatrixCommonElementType_t<LOpResType, ROpResType>;
    using SIMDType = std::conditional_t<
      TMLMatrixIsSameSIMDType_v<LOpResType, ROpResType>,
      TMLMatrixSIMDType_t<LOpResType>, ElementType>;

    explicit TMLDMDMAddExpression(const M1& lhs, const M2& rhs) 
    : m_lhs(~lhs), m_rhs(~rhs) { assert((~lhs).Cols() == (~rhs).Rows()); }

  private:
    // Make copy/move private in order to prevent direct assignment of an expression
    // auto res = m1 + m2 // this is not allowed -> could be dangerous because res is an expression
    // TMLDynamicMatrix<T> res = m1 + m2 // This is allowed
    TMLDMDMAddExpression(const MyT&) = default;
    TMLDMDMAddExpression(MyT&&) noexcept = default;
    MyT& operator=(const MyT&) = default;
    MyT& operator=(MyT&&) noexcept = default;
  public:

    constexpr std::size_t Rows() const noexcept { return (~m_lhs).Rows(); }
    constexpr std::size_t Cols() const noexcept { return (~m_rhs).Cols(); }

    ElementType operator()(std::size_t i, std::size_t j) const noexcept;

    template<typename MT, typename=LOpResType>
    void AssignTo(TMLDenseMatrix<MT>& res) const;

  public:
    template<typename C, typename A, typename B>
    constexpr static bool IsVectorizable_v = 
      TMLMatrixIsDense_v<C> &&
      TMLMatrixIsDense_v<A> &&
      TMLMatrixIsDense_v<B> && 
      TMLMatrixIsVectorized_v<C> &&
      TMLMatrixIsVectorized_v<A> &&
      TMLMatrixIsVectorized_v<B> &&
      TMLMatrixIsSameSIMDType_v<C, A, B>;

    // Selects the right kernel
    template<typename C, typename A, typename B> 
    void ExecuteKernel(TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b) const { DefaultKernel(c, a, b); }
    
    // Addition kernels
    template<typename C, typename A, typename B> 
    static void DefaultKernel(TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b);
    
    const LOpType& m_lhs;
    const ROpType& m_rhs;
  };

  template<typename ML, typename MR, typename=TMLEnableIf_t<TMLBooleanAnd_v<
    TMLMatrixIsDense_v<TMLMatrixExpressionResultType_t<ML>>,
    TMLMatrixIsDense_v<TMLMatrixExpressionResultType_t<MR>>
  >>>
  auto operator+(const ML& lhs, const MR& rhs)
  {
    return TMLDMDMAddExpression<ML, MR>(lhs, rhs);
  }

  template<typename M1, typename M2> typename TMLDMDMAddExpression<M1, M2>::ElementType 
    TMLDMDMAddExpression<M1, M2>::operator()(std::size_t i, std::size_t j) const noexcept
  {
    assert(i < Rows());
    assert(j < Cols());

    return static_cast<ElementType>((~m_lhs)(i, j)) + static_cast<ElementType>((~m_rhs)(i, j));
  }

  template<typename M1, typename M2>
  template<typename MT, typename>
  void TMLDMDMAddExpression<M1, M2>::AssignTo(TMLDenseMatrix<MT>& res) const
  {
    assert((~res).Rows() == (~m_lhs).Rows());
    assert((~res).Cols() == (~m_rhs).Cols());

    const LOpResType& lhs(m_lhs);
    const ROpResType& rhs(m_rhs);

    // no need to check for alias since addition can be performed in-place
    ExecuteKernel(res, lhs, rhs);
  }

  template<typename M1, typename M2>
  template<typename C, typename A, typename B> 
  void TMLDMDMAddExpression<M1, M2>::DefaultKernel(
    TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b)
  {
    for (size_t i = 0; i < (~c).Rows(); i++)
    {
      for (size_t j = 0; j < (~c).Cols(); j++)
      {
        (~c)(i, j) = (~a)(i, j) + (~b)(i, j);
      }
    }
  }

}

#endif
