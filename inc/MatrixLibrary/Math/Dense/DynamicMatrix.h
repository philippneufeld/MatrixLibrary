// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Dense_DynamicMatrix_H_
#define ML_MATH_Dense_DynamicMatrix_H_

// Includes
#include <cassert>
#include <algorithm>

#include "DenseMatrix.h"
#include "DenseMatrixHelper.h"
#include "../SIMD/SIMD.h"

#include "../../Memory/AlignedAlloc.h"

#include "../../QTL/Type.h"
#include "../../QTL/EnableIf.h"
#include "../../QTL/Boolean.h"

namespace ML
{

  //
  // Storage
  //
  namespace Internal
  {
    template<typename Type, std::size_t Al = alignof(Type)>
    class TMLDynamicMatrixStorage
    {
    private:
      using MyT = TMLDynamicMatrixStorage<Type, Al>;
      
    public:
      TMLDynamicMatrixStorage() : m_size(0), m_data(nullptr) {}
      TMLDynamicMatrixStorage(std::size_t size) 
        : m_size(size), m_data(MLAlignedAlloc<Type>(size, Al)) {}
      TMLDynamicMatrixStorage(const MyT& rhs)
        : MyT(rhs.m_size) { if(m_data) std::copy(rhs.m_data, rhs.m_data+m_size, m_data); }
      TMLDynamicMatrixStorage(MyT&& rhs) noexcept
        : m_size(rhs.m_size), m_data(rhs.m_data) { rhs.m_size = 0; rhs.m_data = nullptr; } 
      ~TMLDynamicMatrixStorage() { MLAlignedFree(m_data); m_data = nullptr; m_size = 0; }

      MyT& operator=(const MyT& rhs) { Resize(rhs.m_size, MLNoneType{}); Copy(rhs.m_data); } 
      MyT& operator=(MyT&& rhs) noexcept { std::swap(m_size, rhs.m_size); std::swap(m_data, rhs.m_data); }

      // Size
      void Resize(std::size_t size);
      QM_ALWAYS_INLINE std::size_t GetSize() const noexcept { return m_size; }
      
      // data access (std cpp compliant)
      QM_ALWAYS_INLINE Type* begin() noexcept { return m_data; }
      QM_ALWAYS_INLINE const Type* begin() const noexcept { return m_data; }
      QM_ALWAYS_INLINE const Type* cbegin() const noexcept { return m_data; }
      QM_ALWAYS_INLINE Type* end() noexcept { return m_data + m_size; }
      QM_ALWAYS_INLINE const Type* end() const noexcept { return m_data + m_size; }
      QM_ALWAYS_INLINE const Type* cend() const noexcept { return m_data + m_size; }
      QM_ALWAYS_INLINE Type* data() noexcept { return m_data; }
      QM_ALWAYS_INLINE const Type* data() const noexcept { return m_data; }
      QM_ALWAYS_INLINE Type& operator[](std::size_t i) noexcept { return m_data[i]; }
      QM_ALWAYS_INLINE const Type& operator[](std::size_t i) const noexcept { return m_data[i]; }

    private:
      std::size_t m_size;
      Type* m_data;
    };

    template<typename Type, std::size_t Al>
    void TMLDynamicMatrixStorage<Type, Al>::Resize(std::size_t size)
    {
      if (size != m_size)
      {
        MLAlignedFree(m_data);
        m_data = MLAlignedAlloc<Type>(size, Al);
        m_size = size;
      }
    }

  }

  template <typename ET, bool RowMajor = true, std::size_t maxSIMD = 0xFFFFFFFF>
  class TMLDynamicMatrix : public TMLDenseMatrixHelper<TMLDynamicMatrix<ET, RowMajor, maxSIMD>>
  {
  public:
    // befriend TMLMatrixExpression in order to let it access the SIMD iterator methods
    template<typename T> friend class TMLMatrixExpression;

    // Type aliases
    using MyT = TMLDynamicMatrix<ET, RowMajor, maxSIMD>;
    using TransposeType = TMLDynamicMatrix<ET, !RowMajor, maxSIMD>;
    using ElementType = std::decay_t<ET>;
    using SIMDType = TMLSIMDTypeSelector_t<ElementType, maxSIMD>; 
    
    constexpr static std::size_t SIMDSize = TMLSIMDSize_v<SIMDType>;
  
    // Constructors
    TMLDynamicMatrix() : TMLDynamicMatrix(0, 0) {}
    TMLDynamicMatrix(std::size_t rows, std::size_t cols)
      : TMLDynamicMatrix(rows, cols, MLNoneType{}) { this->SetZero(); }
    explicit TMLDynamicMatrix(MLNoneType) : TMLDynamicMatrix(0, 0, MLNoneType{}) {}
    explicit TMLDynamicMatrix(std::size_t rows, std::size_t cols, MLNoneType);

    // Expression assignment
    template<typename Expr>
    TMLDynamicMatrix(const TMLMatrixExpression<Expr>& expr) 
      : TMLDynamicMatrix((~expr).Rows(), (~expr).Cols(), MLNoneType{}) { (~expr).AssignTo(*this); }
    template<typename Expr>
    TMLDynamicMatrix& operator=(const TMLMatrixExpression<Expr>& expr) { return this->Assign(expr); }
    template<typename Expr>
    TMLDynamicMatrix& Assign(const TMLMatrixExpression<Expr>& expr);

    // copy constructor/assignment operator
    TMLDynamicMatrix(const TMLDynamicMatrix& rhs) noexcept 
      : TMLDynamicMatrix(TMLDMAssignExpression<TMLDynamicMatrix>(rhs)) {}
    TMLDynamicMatrix& operator=(const TMLDynamicMatrix& rhs) noexcept { return this->Assign(rhs); }

    // move constructor/assignment operator
    TMLDynamicMatrix(TMLDynamicMatrix&& rhs) noexcept = default;
    TMLDynamicMatrix& operator=(TMLDynamicMatrix&& rhs) noexcept = default;

    // Matrix assignment
    template<typename MT>
    TMLDynamicMatrix(const TMLDenseMatrix<MT>& rhs)
      : TMLDynamicMatrix(TMLDMAssignExpression<MT>(~rhs)) {}
    template<typename MT>
    TMLDynamicMatrix& operator=(const TMLDenseMatrix<MT>& rhs) noexcept { return this->Assign(~rhs); }
    template<typename MT>
    TMLDynamicMatrix& Assign(const TMLDenseMatrix<MT>& rhs) noexcept { return this->Assign(TMLDMAssignExpression<MT>(~rhs)); }

    // Data access
    QM_ALWAYS_INLINE ElementType &operator()(std::size_t i, std::size_t j) noexcept;
    QM_ALWAYS_INLINE const ElementType &operator()(std::size_t i, std::size_t j) const noexcept;

    // Utility
    void Resize(std::size_t rows, std::size_t cols, MLNoneType);
    void Resize(std::size_t rows, std::size_t cols) { Resize(rows, cols, MLNoneType{}); this->SetZero(); }
    QM_ALWAYS_INLINE std::size_t Rows() const noexcept { return RowMajor ? m_majorCnt : m_minorCnt; }
    QM_ALWAYS_INLINE std::size_t Cols() const noexcept { return RowMajor ? m_minorCnt : m_majorCnt; }
    QM_ALWAYS_INLINE constexpr std::size_t PaddedRows() const noexcept { return RowMajor ? Rows() : m_paddedMinorCnt; }
    QM_ALWAYS_INLINE constexpr std::size_t PaddedCols() const noexcept { return RowMajor ? m_paddedMinorCnt : Cols(); }

    // Alias detection
    template<typename MT> QM_ALWAYS_INLINE TMLEnableIf_t<!TMLMatrixIsDense_v<MT>, bool> 
      IsAlias(const MT& other) { return false; }
    template<typename MT> QM_ALWAYS_INLINE TMLEnableIf_t<TMLMatrixIsDense_v<MT>, bool> 
      IsAlias(const MT& other) { return static_cast<const void*>(this) == static_cast<const void*>(&(~other)); }

    // Load / Store
    SIMDType Load(std::size_t i, std::size_t j) const { return SIMDType::LoadAligned(&((*this)(i, j))); }
    void Store(SIMDType reg, std::size_t i, std::size_t j) { SIMDType::StoreAligned(reg, &((*this)(i, j))); }

  private:
    std::size_t m_majorCnt;
    std::size_t m_minorCnt;
    std::size_t m_paddedMinorCnt;
    Internal::TMLDynamicMatrixStorage<ElementType, alignof(SIMDType)> m_storage;
  };

  template <typename ET, bool RowMajor, std::size_t maxSIMD>
  QM_ALWAYS_INLINE TMLDynamicMatrix<ET, RowMajor, maxSIMD>::TMLDynamicMatrix(std::size_t rows, std::size_t cols, MLNoneType)
    : m_majorCnt(RowMajor ? rows : cols),
      m_minorCnt(RowMajor ? cols : rows),
      m_paddedMinorCnt(m_minorCnt + (SIMDSize - m_minorCnt % SIMDSize) % SIMDSize),
      m_storage(m_majorCnt * m_paddedMinorCnt) {}

  template <typename ET, bool RowMajor, std::size_t maxSIMD>
  void TMLDynamicMatrix<ET, RowMajor, maxSIMD>::Resize(std::size_t rows, std::size_t cols, MLNoneType)
  {
    m_majorCnt = RowMajor ? rows : cols;
    m_minorCnt = RowMajor ? cols : rows;
    m_paddedMinorCnt = m_minorCnt + (SIMDSize - m_minorCnt % SIMDSize) % SIMDSize;
    m_storage.Resize(m_majorCnt * m_paddedMinorCnt);
  }

  template <typename ET, bool RowMajor, std::size_t maxSIMD>
  template<typename Expr> TMLDynamicMatrix<ET, RowMajor, maxSIMD>&
    TMLDynamicMatrix<ET, RowMajor, maxSIMD>::Assign(const TMLMatrixExpression<Expr>& expr)
  {
    this->Resize((~expr).Rows(), (~expr).Cols(), MLNoneType{});
    (~expr).AssignTo(*this);
    return *this;
  }

  template <typename ET, bool RowMajor, std::size_t maxSIMD>
  QM_ALWAYS_INLINE typename TMLDynamicMatrix<ET, RowMajor, maxSIMD>::ElementType& 
    TMLDynamicMatrix<ET, RowMajor, maxSIMD>::operator()(std::size_t i, std::size_t j) noexcept
  {
    assert(i < this->Rows());
    assert(j < this->Cols());

    auto major = RowMajor ? i : j;
    auto minor = RowMajor ? j : i;

    return this->m_storage[major*m_paddedMinorCnt + minor];
  }

  template <typename ET, bool RowMajor, std::size_t maxSIMD>
  QM_ALWAYS_INLINE const typename TMLDynamicMatrix<ET, RowMajor, maxSIMD>::ElementType& 
    TMLDynamicMatrix<ET, RowMajor, maxSIMD>::operator()(std::size_t i, std::size_t j) const noexcept
  {
    assert(i < this->Rows());
    assert(j < this->Cols());

    auto major = RowMajor ? i : j;
    auto minor = RowMajor ? j : i;

    return this->m_storage[major*m_paddedMinorCnt + minor];
  }

  //
  // Traits
  //

  template <typename ET, bool RowMajor, std::size_t maxSIMD>
  struct TMLMatrixRows1<TMLDynamicMatrix<ET, RowMajor, maxSIMD>, void>
    : TMLConstant<std::size_t, MLMatrixDynamicSize_v> {};

  template <typename ET, bool RowMajor, std::size_t maxSIMD>
  struct TMLMatrixCols1<TMLDynamicMatrix<ET, RowMajor, maxSIMD>, void>
    : TMLConstant<std::size_t, MLMatrixDynamicSize_v> {};

  template <typename ET, bool RowMajor, std::size_t maxSIMD>
  struct TMLMatrixIsRowMajor1<TMLDynamicMatrix<ET, RowMajor, maxSIMD>, void>
    : TMLBooleanConstant<RowMajor> {};


  // Arithmetic traits
  template<typename MT1, typename MT2>
  struct TMLMatrixAddResult1<MT1, MT2, 
    TMLEnableIf_t<
      (TMLMatrixIsDynamic_v<MT1> || TMLMatrixIsDynamic_v<MT2>) &&
      TMLMatrixIsElementTypeCompatible_v<MT1, MT2>
    >>
    : TMLType<TMLDynamicMatrix<
        TMLMatrixCommonElementType_t<MT1, MT2>,
        TMLMatrixIsRowMajor_v<MT1>
      >> {};

  template<typename MT1, typename MT2>
  struct TMLMatrixSubResult1<MT1, MT2, 
    TMLEnableIf_t<
      (TMLMatrixIsDynamic_v<MT1> || TMLMatrixIsDynamic_v<MT2>) &&
      TMLMatrixIsElementTypeCompatible_v<MT1, MT2> 
    >>
    : TMLType<TMLDynamicMatrix<
        TMLMatrixCommonElementType_t<MT1, MT2>,
        TMLMatrixIsRowMajor_v<MT1>
      >> {};

  template<typename MT1, typename MT2>
  struct TMLMatrixMulResult1<MT1, MT2, 
    TMLEnableIf_t<
      (TMLMatrixIsDynamic_v<MT1> || TMLMatrixIsDynamic_v<MT2>) &&
      TMLMatrixIsElementTypeCompatible_v<MT1, MT2> 
    >>
    : TMLType<TMLDynamicMatrix<
        TMLMatrixCommonElementType_t<MT1, MT2>,
        TMLMatrixIsRowMajor_v<MT1>
      >> {};

}

#endif
