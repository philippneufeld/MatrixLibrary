// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Expressions_DMSetZero_H_
#define ML_MATH_Expressions_DMSetZero_H_

// Includes
#include "MatrixExpression.h"
#include "../Dense/DenseMatrix.h"

namespace ML
{
  
  template<typename MT>
  class TMLDMSetZeroExpression : public TMLMatrixExpression<TMLDMSetZeroExpression<MT>>
  {
  public:
    using MyT = TMLDMSetZeroExpression;
    using ElementType = TMLMatrixElementType_t<MT>;
    using SIMDType = TMLMatrixSIMDType_t<MT>;
    using ResultType = TMLDecayMatrixType_t<MT>;

    constexpr TMLDMSetZeroExpression(std::size_t rows, std::size_t cols) 
      : m_rows(rows), m_cols(cols) { }

    constexpr std::size_t Rows() const noexcept { return m_rows; }
    constexpr std::size_t Cols() const noexcept { return m_cols; }

    ElementType operator()(std::size_t i, std::size_t j) const noexcept;

    void AssignTo(MT& res) const;
  
  private:
    std::size_t m_rows;
    std::size_t m_cols;
  };


  template<typename MT> typename TMLDMSetZeroExpression<MT>::ElementType 
    TMLDMSetZeroExpression<MT>::operator()(std::size_t i, std::size_t j) const noexcept
  {
    assert(i < Rows());
    assert(j < Cols());
    return 0;
  }

  template<typename MT>
  void TMLDMSetZeroExpression<MT>::AssignTo(MT& res) const
  {
    assert(Rows() == (~res).Rows());
    assert(Cols() == (~res).Cols());

    SIMDType reg = SIMDType::SetZero();

    for (size_t i = 0; i < (~res).Rows(); i++)
    {
      for (size_t j = 0; j < (~res).Cols(); j += TMLSIMDSize_v<SIMDType>)
      {
        (~res).Store(reg, i, j);
      }
    }
  }

}

#endif
