// Copyright 2021, Philipp Neufeld

#ifndef ML_QTL_EnableIf_H_
#define ML_QTL_EnableIf_H_

// Includes
#include "Type.h"

namespace ML
{

  // Defines type depending on the condition
  // If the condition is false, no type is produced and therefore
  // TMLEnableIf<false, T>::type fails
  // This can be used to conditionally enable template expressions because of SFINAE
  template<bool Cond, typename T=void>
  struct TMLEnableIf : TMLType<T> {};
  template<typename T>
  struct TMLEnableIf<false, T> {};
  
  template<bool Cond, typename T=void>
  using TMLEnableIf_t = typename TMLEnableIf<Cond, T>::type;

  // Same as TMLEnableIf but with reversed role of Cond
  template<bool Cond, typename T=void>
  struct TMLDisableIf : TMLEnableIf<!Cond, T> {};

  template<bool Cond, typename T=void>
  using TMLDisableIf_t = typename TMLDisableIf<Cond, T>::type;

}

#endif
