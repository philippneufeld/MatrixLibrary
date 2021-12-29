// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Expressions_DMSet1_H_
#define ML_MATH_Expressions_DMSet1_H_

// Includes
#include "MatrixExpression.h"
#include "../Dense/DenseMatrix.h"

namespace ML
{
  

  template<typename MT>
  class TMLDMSet1Expression : public TMLMatrixExpression<TMLDMSet1Expression<MT>>
  {
  public:
    using MyT = TMLDMSet1Expression<MT>;
    using ElementType = TMLMatrixElementType_t<MT>;
    using SIMDType = TMLMatrixSIMDType_t<MT>;
    using ResultType = TMLDecayMatrixType_t<MT>;

    constexpr TMLDMSet1Expression(ElementType value, std::size_t rows, std::size_t cols) 
      : m_value(value), m_rows(rows), m_cols(cols) { }

    constexpr std::size_t Rows() const noexcept { return m_rows; }
    constexpr std::size_t Cols() const noexcept { return m_cols; }

    ElementType operator()(std::size_t i, std::size_t j) const noexcept;

    void AssignTo(MT& res) const;

  private:
    std::size_t m_rows;
    std::size_t m_cols;
    ElementType m_value;
  };


  template<typename MT> typename TMLDMSet1Expression<MT>::ElementType 
    TMLDMSet1Expression<MT>::operator()(std::size_t i, std::size_t j) const noexcept
  {
    assert(i < Rows());
    assert(j < Cols());
    return m_value;
  }

  template<typename MT>
  void TMLDMSet1Expression<MT>::AssignTo(MT& res) const
  {
    assert(Rows() == (~res).Rows());
    assert(Cols() == (~res).Cols());

    SIMDType reg = SIMDType::Set1(static_cast<ElementType>(this->m_value));
    
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
