// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Expressions_DMDMMul_H_
#define ML_MATH_Expressions_DMDMMul_H_

// Includes
#include <type_traits>
#include <cassert>

#include "MatrixExpression.h"
#include "../Matrix.h"

#include "../../QTL/EnableIf.h"

namespace ML
{

  template<typename M1, typename M2>
  class TMLDMDMMulExpression : public TMLMatrixExpression<TMLDMDMMulExpression<M1, M2>>
  {
    template<typename ML, typename MR, typename>
    friend auto operator*(const ML& lhs, const MR& rhs);

  public:
    using MyT = TMLDMDMMulExpression<M1, M2>;
    using LOpType = TMLDecayCRTP_t<M1>;
    using ROpType = TMLDecayCRTP_t<M2>;
    using LOpResType = TMLMatrixExpressionResultType_t<LOpType>;
    using ROpResType = TMLMatrixExpressionResultType_t<ROpType>;
    using ResultType = TMLMatrixMulResult_t<LOpResType, ROpResType>;

    using ElementType = TMLMatrixCommonElementType_t<LOpResType, ROpResType>;
    using SIMDType = std::conditional_t<
      TMLMatrixIsSameSIMDType_v<LOpResType, ROpResType>,
      TMLMatrixSIMDType_t<LOpResType>, ElementType>;

    explicit TMLDMDMMulExpression(const M1& lhs, const M2& rhs) 
    : m_lhs(~lhs), m_rhs(~rhs) { assert((~lhs).Cols() == (~rhs).Rows()); }

  private:
    // Make copy/move private in order to prevent direct assignment of an expression
    // auto res = m1 * m2 // this is not allowed -> could be dangerous because res is an expression
    // TMLDynamicMatrix<T> res = m1 * m2 // This is allowed
    TMLDMDMMulExpression(const MyT&) = default;
    TMLDMDMMulExpression(MyT&&) noexcept = default;
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
    template<typename C, typename A, typename B> TMLEnableIf_t<!IsVectorizable_v<C, A, B>, void> 
      ExecuteKernel(TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b) const { DefaultKernel(c, a, b); }
    template<typename C, typename A, typename B> TMLEnableIf_t<IsVectorizable_v<C, A, B>, void>
      ExecuteKernel(TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b) const { VectorizedKernel(c, a, b); }

    // Multplication kernels
    template<typename C, typename A, typename B> 
    static void DefaultKernel(TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b);
    template<typename C, typename A, typename B, typename=TMLEnableIf_t<IsVectorizable_v<C, A, B>>>
    static void VectorizedKernel(TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b);
    
    template<std::size_t regsA, std::size_t regsB, typename C, typename A, typename B, typename=TMLEnableIf_t<IsVectorizable_v<C, A, B>>>
    QM_ALWAYS_INLINE static void VectorizedSubKernelRRR(
      TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b, std::size_t ro, std::size_t co);

    const LOpType& m_lhs;
    const ROpType& m_rhs;
  };

  template<typename ML, typename MR, typename=TMLEnableIf_t<TMLBooleanAnd_v<
    TMLMatrixIsDense_v<TMLMatrixExpressionResultType_t<ML>>,
    TMLMatrixIsDense_v<TMLMatrixExpressionResultType_t<MR>>
  >>>
  auto operator*(const ML& lhs, const MR& rhs)
  {
    return TMLDMDMMulExpression<ML, MR>(lhs, rhs);
  }

  template<typename M1, typename M2> typename TMLDMDMMulExpression<M1, M2>::ElementType 
    TMLDMDMMulExpression<M1, M2>::operator()(std::size_t i, std::size_t j) const noexcept
  {
    assert(i < Rows());
    assert(j < Cols());
    
    ElementType res = 0;
    for (size_t k = 0; k < (~m_lhs).Cols(); k++)
      res += static_cast<ElementType>((~m_lhs)(i, k)) * static_cast<ElementType>((~m_rhs)(k, j));
    return res;
  }

  template<typename M1, typename M2>
  template<typename MT, typename>
  void TMLDMDMMulExpression<M1, M2>::AssignTo(TMLDenseMatrix<MT>& res) const
  {
    assert((~res).Rows() == (~m_lhs).Rows());
    assert((~res).Cols() == (~m_rhs).Cols());

    const LOpResType& lhs(m_lhs);
    const ROpResType& rhs(m_rhs);

    if ((~res).IsAlias(lhs) || (~res).IsAlias(rhs))
    {
      using TmpType = decltype(~res);
      auto tmp = TmpType(~res);
      ExecuteKernel(tmp, lhs, rhs);
      (~res).Assign(tmp);
    }
    else
    {
      ExecuteKernel(res, lhs, rhs);
    }
  }

  template<typename M1, typename M2>
  template<typename C, typename A, typename B> 
  void TMLDMDMMulExpression<M1, M2>::DefaultKernel(
    TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b)
  {
    (~c).SetZero();
    for (size_t i = 0; i < (~a).Rows(); i++)
    {
      for (size_t k = 0; k < (~a).Cols(); k++)
      {
        auto a_ik = static_cast<ElementType>((~a)(i, k));
        for (size_t j = 0; j < (~b).Cols(); j++)
        {
          (~c)(i, j) += a_ik * static_cast<ElementType>((~b)(k, j));
        }
      }
    }
  }

  template<typename M1, typename M2>
  template<typename C, typename A, typename B, typename> 
  void TMLDMDMMulExpression<M1, M2>::VectorizedKernel(
    TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b)
  {
    constexpr auto simdSize = TMLSIMDSize_v<SIMDType>;

    if (TMLMatrixIsRowMajor_v<C> && TMLMatrixIsRowMajor_v<A> && TMLMatrixIsRowMajor_v<B>)
    {
      // Cascade the different kernel sizes
      std::size_t j = 0;
      for(; (j + 4*simdSize) <= (~c).PaddedCols(); j += 4*simdSize)
      {
        std::size_t i = 0;
        for (; (i + 3) <= (~c).Rows(); i += 3) { VectorizedSubKernelRRR<3, 4>(c, a, b, i, j); }
        for (; (i + 2) <= (~c).Rows(); i += 2) { VectorizedSubKernelRRR<2, 4>(c, a, b, i, j); }
        if (i < (~c).Rows()) { VectorizedSubKernelRRR<1, 4>(c, a, b, i, j); }
      }
      for(; (j + 3*simdSize) <= (~c).PaddedCols(); j += 3*simdSize)
      {
        std::size_t i = 0;
        for (; (i + 4) <= (~c).Rows(); i += 4) { VectorizedSubKernelRRR<4, 3>(c, a, b, i, j); }
        for (; (i + 2) <= (~c).Rows(); i += 2) { VectorizedSubKernelRRR<2, 3>(c, a, b, i, j); }
        if (i < (~c).Rows()) { VectorizedSubKernelRRR<1, 3>(c, a, b, i, j); }
      }
      for(; (j + 2*simdSize) <= (~c).PaddedCols(); j += 2*simdSize)
      {
        std::size_t i = 0;
        for (; (i + 2) <= (~c).Rows(); i += 2) { VectorizedSubKernelRRR<2, 2>(c, a, b, i, j); }
        if (i < (~c).Rows()) { VectorizedSubKernelRRR<1, 2>(c, a, b, i, j); }
      }
      for(; (j + simdSize) <= (~c).PaddedCols(); j += simdSize)
      {
        std::size_t i = 0;
        for (; (i + 8) <= (~c).Rows(); i += 8) { VectorizedSubKernelRRR<8, 1>(c, a, b, i, j); }
        for (; (i + 4) <= (~c).Rows(); i += 4) { VectorizedSubKernelRRR<4, 1>(c, a, b, i, j); }
        for (; (i + 2) <= (~c).Rows(); i += 2) { VectorizedSubKernelRRR<2, 1>(c, a, b, i, j); }
        for (; i < (~c).Rows(); i++) { VectorizedSubKernelRRR<1, 1>(c, a, b, i, j); }
      }
    }
    else
    {
      DefaultKernel(c, a, b);
    } 
  }

  template<typename M1, typename M2>
  template<std::size_t regsA, std::size_t regsB, typename C, typename A, typename B, typename> 
  QM_ALWAYS_INLINE void TMLDMDMMulExpression<M1, M2>::VectorizedSubKernelRRR(
    TMLDenseMatrix<C>& c, const TMLDenseMatrix<A>& a, const TMLDenseMatrix<B>& b, std::size_t ro, std::size_t co)
  {
    if ((~a).Rows() > 0) 
    {
      SIMDType csum[regsA][regsB];

      // First p
      std::size_t p = 0;
      MLConstexprFor<std::size_t, 0, regsB, 1>([&](auto bi) {
        auto bb = (~b).Load(p, co + bi * TMLSIMDSize_v<SIMDType>);
        MLConstexprFor<std::size_t, 0, regsA, 1>([&](auto ai) {
          auto aa = SIMDType::Set1((~a)(ro + ai, p));
            csum[ai][bi] = aa * bb;
        });
      });

      // Rest of ps
      for (++p; p < (~a).Rows(); p++) {
        MLConstexprFor<std::size_t, 0, regsB, 1>([&](auto bi) {
          auto bb = (~b).Load(p, co + bi * TMLSIMDSize_v<SIMDType>);
          MLConstexprFor<std::size_t, 0, regsA, 1>([&](auto ai) {
            auto aa = SIMDType::Set1((~a)(ro + ai, p));
              csum[ai][bi] = MLSIMDFmadd(aa, bb, csum[ai][bi]);
          });
        });
      }

      // Accumulate the results into C.
      for (std::size_t ai = 0; ai < regsA; ai++) {
        for (std::size_t bi = 0; bi < regsB; bi++) {
          (~c).Store(csum[ai][bi], ro + ai, co + bi * TMLSIMDSize_v<SIMDType>);
        }
      }
    }
  }

}

#endif
