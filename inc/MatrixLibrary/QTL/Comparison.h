// Copyright 2021, Philipp Neufeld

#ifndef ML_QTL_Comparison_H_
#define ML_QTL_Comparison_H_

// Includes
#include "Boolean.h"

namespace ML
{

  // Check if v1 is equal to v2
  template<typename T, T v1, T v2>
  struct TMLEqual : TMLBooleanConstant<(v1 == v2)> {};

  template<typename T, T v1, T v2>
  constexpr bool TMLEqual_v = TMLEqual<T, v1, v2>::value;

  // Check if v1 is not equal to v2
  template<typename T, T v1, T v2>
  struct TMLNotEqual : TMLBooleanConstant<(v1 != v2)> {};

  template<typename T, T v1, T v2>
  constexpr bool TMLNotEqual_v = TMLNotEqual<T, v1, v2>::value;

  // Check if v1 is less than v2
  template<typename T, T v1, T v2>
  struct TMLLess : TMLBooleanConstant<(v1 < v2)> {};

  template<typename T, T v1, T v2>
  constexpr bool TMLLess_v = TMLLess<T, v1, v2>::value;

  // Check if v1 is less than or equal to v2
  template<typename T, T v1, T v2>
  struct TMLLessEqual : TMLBooleanConstant<(v1 <= v2)> {};

  template<typename T, T v1, T v2>
  constexpr bool TMLLessEqual_v = TMLLessEqual<T, v1, v2>::value;

  // Check if v1 is greater than v2
  template<typename T, T v1, T v2>
  struct TMLGreater : TMLBooleanConstant<!(v1 <= v2)> {};

  template<typename T, T v1, T v2>
  constexpr bool TMLGreater_v = TMLGreater<T, v1, v2>::value;

  // Check if v1 is less than or equal to v2
  template<typename T, T v1, T v2>
  struct TMLGreaterEqual : TMLBooleanConstant<!(v1 < v2)> {};

  template<typename T, T v1, T v2>
  constexpr bool TMLGreaterEqual_v = TMLGreaterEqual<T, v1, v2>::value;

}

#endif
