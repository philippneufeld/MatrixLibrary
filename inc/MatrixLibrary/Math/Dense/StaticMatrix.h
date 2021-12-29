// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Dense_StaticMatrix_H_
#define ML_MATH_Dense_StaticMatrix_H_

// Includes
#include <cassert>
#include <algorithm>

#include "DenseMatrix.h"
#include "DenseMatrixHelper.h"
#include "../SIMD/SIMD.h"

#include "../../QTL/Type.h"
#include "../../QTL/EnableIf.h"
#include "../../QTL/Boolean.h"
#include "../../QTL/Comparison.h"

#include "../../Memory/AlignedAlloc.h"

namespace ML
{

  //
  // Storage
  //
  namespace Internal
  {
    template<typename Type, std::size_t N, std::size_t Al = alignof(Type), typename=void>
    class TMLStaticMatrixStorage
    {
    private:
      using MyT = TMLStaticMatrixStorage<Type, N, Al>;

    public:
      TMLStaticMatrixStorage() {}; // no init for m_data!
      TMLStaticMatrixStorage(const TMLStaticMatrixStorage&) = default;
      TMLStaticMatrixStorage(TMLStaticMatrixStorage&&) noexcept = default;

      TMLStaticMatrixStorage& operator=(const TMLStaticMatrixStorage&) = default;
      TMLStaticMatrixStorage& operator=(TMLStaticMatrixStorage&&) noexcept = default;

      // Size
      QM_ALWAYS_INLINE constexpr std::size_t GetSize() const noexcept { return N; }
      
      // data access (std cpp compliant)
      QM_ALWAYS_INLINE constexpr Type* begin() noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr const Type* begin() const noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr const Type* cbegin() const noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr Type* end() noexcept { return m_data + N; }
      QM_ALWAYS_INLINE constexpr const Type* end() const noexcept { return m_data + N; }
      QM_ALWAYS_INLINE constexpr const Type* cend() const noexcept { return m_data + N; }
      QM_ALWAYS_INLINE constexpr Type* data() noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr const Type* data() const noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr Type& operator[](std::size_t i) noexcept { return m_data[i]; }
      QM_ALWAYS_INLINE constexpr const Type& operator[](std::size_t i) const noexcept { return m_data[i]; }

    private:
      alignas(Al) Type m_data[N];
    };

    // Starting from sizes greated than 16kB the matrix is allocated dynamically
    template<typename Type, std::size_t N, std::size_t Al>
    class TMLStaticMatrixStorage<Type, N, Al, 
      TMLEnableIf_t<TMLGreaterEqual_v<std::size_t, sizeof(Type) * N, 16 * 1024>>>
    {
    private:
      using MyT = TMLStaticMatrixStorage<Type, N, Al>;

    public:
      TMLStaticMatrixStorage() : m_data(MLAlignedAlloc<Type>(N, Al)) {}; // no init for m_data!
      ~TMLStaticMatrixStorage() { MLAlignedFree(m_data); m_data = nullptr; }

      TMLStaticMatrixStorage(const MyT& rhs) 
        : m_data(MLAlignedAlloc<Type>(N, Al)) { std::copy(rhs.m_data, rhs.m_data + N, m_data); }
      TMLStaticMatrixStorage(MyT&& rhs) noexcept : m_data(rhs.m_data) { rhs.m_data = nullptr; }

      MyT& operator=(const MyT& rhs) { std::copy(rhs.m_data, rhs.m_data + N, m_data); }
      MyT& operator=(MyT&& rhs) noexcept { std::swap(m_data, rhs.m_data); return *this; }

      // Size
      QM_ALWAYS_INLINE constexpr std::size_t GetSize() const noexcept { return N; }
      
      // data access (std cpp compliant)
      QM_ALWAYS_INLINE constexpr Type* begin() noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr const Type* begin() const noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr const Type* cbegin() const noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr Type* end() noexcept { return m_data + N; }
      QM_ALWAYS_INLINE constexpr const Type* end() const noexcept { return m_data + N; }
      QM_ALWAYS_INLINE constexpr const Type* cend() const noexcept { return m_data + N; }
      QM_ALWAYS_INLINE constexpr Type* data() noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr const Type* data() const noexcept { return m_data; }
      QM_ALWAYS_INLINE constexpr Type& operator[](std::size_t i) noexcept { return m_data[i]; }
      QM_ALWAYS_INLINE constexpr const Type& operator[](std::size_t i) const noexcept { return m_data[i]; }

    private:
      Type* m_data;
    };
  }

  //
  // Memory layout helper
  //
  namespace Internal
  {
    template<typename ST, std::size_t MajCnt, std::size_t MinCnt, typename=void>
    class TMLStaticMatrixMemoryLayout
    {
    private:
      using SIMDType = TMLDecaySIMDType_t<ST>;
      using ElementType = TMLSIMDElementType_t<SIMDType>;
      constexpr static std::size_t SIMDSize_v = TMLSIMDSize_v<SIMDType>;
    
    public:
      constexpr static std::size_t PaddedMinorCnt_v = MinCnt + (SIMDSize_v - MinCnt % SIMDSize_v) % SIMDSize_v;
      constexpr static std::size_t Alignment_v = alignof(SIMDType);
      constexpr static std::size_t PaddedSize_v = MajCnt * PaddedMinorCnt_v;

      constexpr static std::size_t CalcIndex(std::size_t major, std::size_t minor) { return major*PaddedMinorCnt_v + minor; }
    };
  }

  template <typename ET, std::size_t N, std::size_t M, bool RowMajor = true, std::size_t maxSIMD = RowMajor ? M : N>
  class TMLStaticMatrix : public TMLDenseMatrixHelper<TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>>
  {
    static_assert(N != MLMatrixDynamicSize_v && M != MLMatrixDynamicSize_v, "Invalid row or column size");
  public:
    // befriend TMLMatrixExpression in order to let it access the SIMD iterator methods
    template<typename T> friend class TMLMatrixExpression;

    using MyT = TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>;
    using TransposeType = TMLStaticMatrix<ET, M, N, !RowMajor, maxSIMD>;
    using ElementType = std::decay_t<ET>;
    using SIMDType = TMLSIMDTypeSelector_t<ElementType, maxSIMD>; 
    
    using MemoryLayout = Internal::TMLStaticMatrixMemoryLayout<SIMDType, RowMajor ? N : M, RowMajor ? M : N>;
    
    // Constructors
    TMLStaticMatrix() noexcept { this->SetZero(); }
    explicit TMLStaticMatrix(MLNoneType) noexcept { } // non-initialization constructor
    
    // Expression assignment
    template<typename Expr>
    TMLStaticMatrix(const TMLMatrixExpression<Expr>& expr) noexcept 
      : TMLStaticMatrix(MLNoneType{}) { this->Assign(expr); }
    template<typename Expr>
    TMLStaticMatrix& operator=(const TMLMatrixExpression<Expr>& expr) noexcept { return this->Assign(expr); }
    template<typename Expr>
    TMLStaticMatrix& Assign(const TMLMatrixExpression<Expr>& expr) noexcept;

    // copy constructor/assignment operator
    TMLStaticMatrix(const TMLStaticMatrix& rhs) noexcept 
      : TMLStaticMatrix(TMLDMAssignExpression<TMLStaticMatrix>(rhs)) {}
    TMLStaticMatrix& operator=(const TMLStaticMatrix& rhs) noexcept { return this->Assign(rhs); }

    // move constructor/assignment operator
    TMLStaticMatrix(TMLStaticMatrix&& rhs) noexcept = default;
    TMLStaticMatrix& operator=(TMLStaticMatrix&& rhs) noexcept = default;

    // Matrix assignment
    template<typename MT>
    TMLStaticMatrix(const TMLDenseMatrix<MT>& rhs)
      : TMLStaticMatrix(TMLDMAssignExpression<MT>(~rhs)) {}
    template<typename MT>
    TMLStaticMatrix& operator=(const TMLDenseMatrix<MT>& rhs) noexcept { return this->Assign(~rhs); }
    template<typename MT>
    TMLStaticMatrix& Assign(const TMLDenseMatrix<MT>& rhs) noexcept { return this->Assign(TMLDMAssignExpression<MT>(~rhs)); }

    // data access
    QM_ALWAYS_INLINE ElementType& operator()(std::size_t i, std::size_t j) noexcept;
    QM_ALWAYS_INLINE const ElementType& operator()(std::size_t i, std::size_t j) const noexcept;

    // Utility
    QM_ALWAYS_INLINE constexpr std::size_t Rows() const noexcept { return N; }
    QM_ALWAYS_INLINE constexpr std::size_t Cols() const noexcept { return M; }
    QM_ALWAYS_INLINE constexpr std::size_t PaddedRows() const noexcept { return RowMajor ? Rows() : MemoryLayout::PaddedMinorCnt_v; }
    QM_ALWAYS_INLINE constexpr std::size_t PaddedCols() const noexcept { return RowMajor ? MemoryLayout::PaddedMinorCnt_v : Cols(); }
    
    // Alias detection
    template<typename MT> QM_ALWAYS_INLINE TMLEnableIf_t<!TMLMatrixIsDense_v<MT>, bool> 
      IsAlias(const MT& other) { return false; }
    template<typename MT> QM_ALWAYS_INLINE TMLEnableIf_t<TMLMatrixIsDense_v<MT>, bool> 
      IsAlias(const MT& other) { return static_cast<const void*>(this) == static_cast<const void*>(&(~other)); }
    
    // Load / Store
    SIMDType Load(std::size_t i, std::size_t j) const { return SIMDType::LoadAligned(&((*this)(i, j))); }
    void Store(SIMDType reg, std::size_t i, std::size_t j) { SIMDType::StoreAligned(reg, &((*this)(i, j))); }

  private:
    Internal::TMLStaticMatrixStorage<ElementType, MemoryLayout::PaddedSize_v, MemoryLayout::Alignment_v> m_storage;
  };
  
  template <typename ET, std::size_t N, std::size_t M, bool RowMajor, std::size_t maxSIMD>
  template<typename Expr> TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>&
    TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>::Assign(const TMLMatrixExpression<Expr>& expr) noexcept
  {
    assert((~expr).Rows() == this->Rows());
    assert((~expr).Cols() == this->Cols());
    (~expr).AssignTo(~(*this));
    return *this;
  }

  template <typename ET, std::size_t N, std::size_t M, bool RowMajor, std::size_t maxSIMD>
  QM_ALWAYS_INLINE typename TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>::ElementType& 
    TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>::operator()(std::size_t i, std::size_t j) noexcept
  {
    assert(i < this->Rows());
    assert(j < this->Cols());

    auto major = RowMajor ? i : j;
    auto minor = RowMajor ? j : i;

    return this->m_storage[MemoryLayout::CalcIndex(major, minor)];
  }

  template <typename ET, std::size_t N, std::size_t M, bool RowMajor, std::size_t maxSIMD>
  QM_ALWAYS_INLINE const typename TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>::ElementType& 
    TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>::operator()(std::size_t i, std::size_t j) const noexcept
  {
    assert(i < this->Rows());
    assert(j < this->Cols());

    auto major = RowMajor ? i : j;
    auto minor = RowMajor ? j : i;

    return this->m_storage[MemoryLayout::CalcIndex(major, minor)];
  }

  //
  // Traits
  //

  template<typename ET, std::size_t N, std::size_t M, bool RowMajor, std::size_t maxSIMD>
  struct TMLMatrixRows1<TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>, void>
    : TMLConstant<std::size_t, N> {};

  template<typename ET, std::size_t N, std::size_t M, bool RowMajor, std::size_t maxSIMD>
  struct TMLMatrixCols1<TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>, void>
    : TMLConstant<std::size_t, M> {};

  template<typename ET, std::size_t N, std::size_t M, bool RowMajor, std::size_t maxSIMD>
  struct TMLMatrixIsRowMajor1<TMLStaticMatrix<ET, N, M, RowMajor, maxSIMD>, void>
    : TMLBooleanConstant<RowMajor> {};


  // Arithmetic traits
  template<typename MT1, typename MT2>
  struct TMLMatrixAddResult1<MT1, MT2, 
    TMLEnableIf_t<
      TMLMatrixIsStatic_v<MT1> && 
      TMLMatrixIsStatic_v<MT2> &&
      TMLMatrixIsElementTypeCompatible_v<MT1, MT2> &&
      TMLMatrixRows_v<MT1> == TMLMatrixRows_v<MT2> &&
      TMLMatrixCols_v<MT1> == TMLMatrixCols_v<MT2>
    >>
    : TMLType<TMLStaticMatrix<
        TMLMatrixCommonElementType_t<MT1, MT2>, 
        TMLMatrixRows_v<MT1>, 
        TMLMatrixCols_v<MT1>,
        TMLMatrixIsRowMajor_v<MT1>
      >> {};
  

  template<typename MT1, typename MT2>
  struct TMLMatrixSubResult1<MT1, MT2, 
    TMLEnableIf_t<
      TMLMatrixIsStatic_v<MT1> && 
      TMLMatrixIsStatic_v<MT2> &&
      TMLMatrixIsElementTypeCompatible_v<MT1, MT2> &&
      TMLMatrixRows_v<MT1> == TMLMatrixRows_v<MT2> &&
      TMLMatrixCols_v<MT1> == TMLMatrixCols_v<MT2>
    >>
    : TMLType<TMLStaticMatrix<
        TMLMatrixCommonElementType_t<MT1, MT2>, 
        TMLMatrixRows_v<MT1>, 
        TMLMatrixCols_v<MT1>,
        TMLMatrixIsRowMajor_v<MT1>
      >> {};

  template<typename MT1, typename MT2>
  struct TMLMatrixMulResult1<MT1, MT2, 
    TMLEnableIf_t<
      TMLMatrixIsStatic_v<MT1> && 
      TMLMatrixIsStatic_v<MT2> &&
      TMLMatrixIsElementTypeCompatible_v<MT1, MT2> &&
      TMLMatrixCols_v<MT1> == TMLMatrixRows_v<MT2>
    >>
    : TMLType<TMLStaticMatrix<
        TMLMatrixCommonElementType_t<MT1, MT2>, 
        TMLMatrixRows_v<MT1>, 
        TMLMatrixCols_v<MT2>,
        TMLMatrixIsRowMajor_v<MT1>
      >> {};

}

#endif
