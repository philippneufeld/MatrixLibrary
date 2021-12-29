// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_Matrix_H_
#define ML_MATH_Matrix_H_

// Includes
#include <type_traits>

#include "MathPrerequisites.h"
#include "SIMD/SIMD.h"
#include "../QTL/TypeList.h"
#include "../QTL/Type.h"
#include "../QTL/Boolean.h"
#include "../QTL/EnableIf.h"


namespace ML
{

  template<typename MT>
  struct TMLMatrix : TMLCRTP<MT> {};

  //
  // Traits
  //

  // Trait that checks if type is a matrix
  template<typename MT>
  struct TMLIsMatrix : TMLIsCRTP<MT, TMLMatrix> {};
  
  template<typename MT>
  constexpr bool TMLIsMatrix_v = TMLIsMatrix<MT>::value;

  // Get the inner CRTP type
  template<typename MT, typename=void>
  struct TMLDecayMatrixType;
  template<typename MT>
  struct TMLDecayMatrixType<MT, TMLEnableIf_t<TMLIsMatrix_v<MT>>> 
    : TMLDecayCRTP<MT> {};
  
  template<typename MT>
  using TMLDecayMatrixType_t = typename TMLDecayMatrixType<MT>::type;

  // Get element type of a matrix
  template<typename MT, typename=void>
  struct TMLMatrixElementType;
  template<typename MT>
  struct TMLMatrixElementType<MT, TMLEnableIf_t<TMLIsMatrix_v<MT> && TMLTrue_v<typename TMLDecayMatrixType_t<MT>::ElementType>>>
    : TMLType<typename TMLDecayMatrixType_t<MT>::ElementType> {};

  template<typename MT>
  using TMLMatrixElementType_t = typename TMLMatrixElementType<MT>::type;

  // Get SIMD type of a matrix
  template<typename MT, typename=void>
  struct TMLMatrixSIMDType;
  template<typename MT>
  struct TMLMatrixSIMDType<MT, TMLEnableIf_t<TMLIsMatrix_v<MT> && TMLTrue_v<typename TMLDecayMatrixType_t<MT>::SIMDType>>>
    : TMLType<typename TMLDecayMatrixType_t<MT>::SIMDType> {};

  template<typename MT>
  using TMLMatrixSIMDType_t = typename TMLMatrixSIMDType<MT>::type;

  // Check compatibility of matrix element types
  namespace Internal
  {
    template<typename TL, typename=void>
    struct TMLMatrixIsElementTypeCompatible_impl : std::false_type {};
    template<typename MT1, typename... MTs>
    struct TMLMatrixIsElementTypeCompatible_impl<
      TMLTypelist<MT1, MTs...>, 
      TMLEnableIf_t<TMLTrue_v<std::common_type_t<TMLMatrixElementType_t<MT1>, TMLMatrixElementType_t<MTs>...>>>
    > : std::true_type {};
  }

  template<typename MT1, typename... MTs>
  struct TMLMatrixIsElementTypeCompatible 
    : Internal::TMLMatrixIsElementTypeCompatible_impl<TMLTypelist<MT1, MTs...>> {};
  
  template<typename MT1, typename... MTs>
  constexpr bool TMLMatrixIsElementTypeCompatible_v = TMLMatrixIsElementTypeCompatible<MT1, MTs...>::value;

  // Compatible element type of matrix types
  namespace Internal
  {
    template<typename MTL, typename=void>
    struct TMLMatrixCommonElementType_impl;
    template<typename MT1, typename... MTs>
    struct TMLMatrixCommonElementType_impl<
      TMLTypelist<MT1, MTs...>, 
      TMLEnableIf_t<TMLMatrixIsElementTypeCompatible_v<MT1, MTs...>>>
      : TMLType<std::common_type_t<TMLMatrixElementType_t<MTs>...>> {};
  }

  template<typename MT1, typename... MTs>
  struct TMLMatrixCommonElementType 
    : Internal::TMLMatrixCommonElementType_impl<TMLTypelist<MT1, MTs...>> {};
  
  template<typename MT1, typename... MTs>
  using TMLMatrixCommonElementType_t = typename TMLMatrixCommonElementType<MT1, MTs...>::type;

  // Check if matrices have the same element type
  namespace Internal
  {
    template<typename MTL, typename=void>
    struct TMLMatrixIsSameElementType_impl : std::false_type {};
    template<typename MT1, typename MT2, typename... MTs>
    struct TMLMatrixIsSameElementType_impl<TMLTypelist<MT1, MT2, MTs...>, 
      TMLEnableIf_t<
        std::is_same<TMLMatrixElementType_t<MT1>, TMLMatrixElementType_t<MT2>>::value &&
        TMLMatrixIsSameElementType_impl<TMLTypelist<MT2, MTs...>>::value
      >> : std::true_type {};
    template<typename MT1>
    struct TMLMatrixIsSameElementType_impl<TMLTypelist<MT1>, void> : std::true_type {};
  }

  template<typename MT1, typename MT2, typename... MTs>
  struct TMLMatrixIsSameElementType 
    : Internal::TMLMatrixIsSameElementType_impl<TMLTypelist<MT1, MT2, MTs...>> {};
  
  template<typename MT1, typename MT2, typename... MTs>
  constexpr bool TMLMatrixIsSameElementType_v = TMLMatrixIsSameElementType<MT1, MT2, MTs...>::value;

  // Check if matrices have the same SIMD type
  namespace Internal
  {
    template<typename MTL, typename=void>
    struct TMLMatrixIsSameSIMDType_impl : std::false_type {};
    template<typename MT1, typename MT2, typename... MTs>
    struct TMLMatrixIsSameSIMDType_impl<TMLTypelist<MT1, MT2, MTs...>, 
      TMLEnableIf_t<
        std::is_same<TMLMatrixSIMDType_t<MT1>, TMLMatrixSIMDType_t<MT2>>::value &&
        TMLMatrixIsSameSIMDType_impl<TMLTypelist<MT2, MTs...>>::value
      >> : std::true_type {};
    template<typename MT1>
    struct TMLMatrixIsSameSIMDType_impl<TMLTypelist<MT1>, void> : std::true_type {};
  }

  template<typename MT1, typename MT2, typename... MTs>
  struct TMLMatrixIsSameSIMDType 
    : Internal::TMLMatrixIsSameSIMDType_impl<TMLTypelist<MT1, MT2, MTs...>> {};
  
  template<typename MT1, typename MT2, typename... MTs>
  constexpr bool TMLMatrixIsSameSIMDType_v = TMLMatrixIsSameSIMDType<MT1, MT2, MTs...>::value;

  // Check if matrix type uses vectorization
  template<typename MT, typename=void>
  struct TMLMatrixIsVectorized;
  template<typename MT>
  struct TMLMatrixIsVectorized<MT, TMLEnableIf_t<TMLIsMatrix<MT>::value>>
    : TMLBooleanConstant<TMLSIMDSize_v<TMLMatrixSIMDType_t<MT>> != 1> {};

  template<typename MT>
  constexpr bool TMLMatrixIsVectorized_v = TMLMatrixIsVectorized<MT>::value;

  // Dynamic size (for TMLMatrixRows and TMLMatrixCols)
  constexpr std::size_t MLMatrixDynamicSize_v = static_cast<std::size_t>(-1);

  // Determines rows of a matrix (MLMatrixDynamicSize_v if it is dynamic)
  template<typename MT, typename=void>
  struct TMLMatrixRows1;
  template<typename MT>
  struct TMLMatrixRows
    : TMLMatrixRows1<TMLDecayMatrixType_t<MT>> {};
  
  template<typename MT>
  constexpr std::size_t TMLMatrixRows_v = TMLMatrixRows<MT>::value;

  // Determines columns of a matrix (MLMatrixDynamicSize_v if it is dynamic)
  template<typename MT, typename=void>
  struct TMLMatrixCols1;
  template<typename MT>
  struct TMLMatrixCols
    : TMLMatrixCols1<TMLDecayMatrixType_t<MT>> {};

  template<typename MT>
  constexpr std::size_t TMLMatrixCols_v = TMLMatrixCols<MT>::value;

  // Checks if matrix is statically sized
  template<typename MT, typename=void>
  struct TMLMatrixIsStatic : std::false_type {};
  template<typename MT>
  struct TMLMatrixIsStatic<MT, 
    TMLEnableIf_t<
      TMLIsMatrix_v<MT> &&
      TMLMatrixRows_v<MT> != MLMatrixDynamicSize_v && 
      TMLMatrixCols_v<MT> != MLMatrixDynamicSize_v
    >> : std::true_type {};

  template<typename MT>
  constexpr bool TMLMatrixIsStatic_v = TMLMatrixIsStatic<MT>::value;

  // Checks if matrix is dynamically sized
  template<typename MT, typename=void>
  struct TMLMatrixIsDynamic : std::false_type {};
  template<typename MT>
  struct TMLMatrixIsDynamic<MT, 
    TMLEnableIf_t<
      TMLIsMatrix_v<MT> &&
      (TMLMatrixRows_v<MT> == MLMatrixDynamicSize_v || 
      TMLMatrixCols_v<MT> == MLMatrixDynamicSize_v)
    >> : std::true_type {};
  
  template<typename MT>
  constexpr bool TMLMatrixIsDynamic_v = TMLMatrixIsDynamic<MT>::value;

  // Check for storage order (row major)
  template<typename MT, typename=void>
  struct TMLMatrixIsRowMajor1;
  template<typename MT>
  struct TMLMatrixIsRowMajor 
    : TMLMatrixIsRowMajor1<TMLDecayMatrixType_t<MT>> {};
  
  template<typename MT>
  constexpr bool TMLMatrixIsRowMajor_v = TMLMatrixIsRowMajor<MT>::value;
  
  // Check for storage order (column major)
  template<typename MT>
  struct TMLMatrixIsColumnMajor 
    : TMLBooleanConstant<!TMLMatrixIsRowMajor<MT>::value> {};
  
  template<typename MT>
  constexpr bool TMLMatrixIsColumnMajor_v = TMLMatrixIsColumnMajor<MT>::value;
  
  // Addition result
  template<typename MT1, typename MT2, typename=void>
  struct TMLMatrixAddResult1;
  template<typename MT1, typename MT2>
  struct TMLMatrixAddResult 
    : TMLMatrixAddResult1<TMLDecayMatrixType_t<MT1>, TMLDecayMatrixType_t<MT2>> {};
  
  template<typename MT1, typename MT2>
  using TMLMatrixAddResult_t = typename TMLMatrixAddResult<MT1, MT2>::type;

  // Subtraction result
  template<typename MT1, typename MT2, typename=void>
  struct TMLMatrixSubResult1;
  template<typename MT1, typename MT2>
  struct TMLMatrixSubResult 
    : TMLMatrixSubResult1<TMLDecayMatrixType_t<MT1>, TMLDecayMatrixType_t<MT2>> {};

  template<typename MT1, typename MT2>
  using TMLMatrixSubResult_t = typename TMLMatrixSubResult<MT1, MT2>::type;

  // Multiplication result
  template<typename MT1, typename MT2, typename=void>
  struct TMLMatrixMulResult1;
  template<typename MT1, typename MT2>
  struct TMLMatrixMulResult
    : TMLMatrixMulResult1<TMLDecayMatrixType_t<MT1>, TMLDecayMatrixType_t<MT2>> {};
  
  template<typename MT1, typename MT2>
  using TMLMatrixMulResult_t = typename TMLMatrixMulResult<MT1, MT2>::type;

}

#include "Dense/DenseMatrix.h"

#endif