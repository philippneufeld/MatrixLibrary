// Copyright 2021, Philipp Neufeld

#ifndef ML_QTL_Type_H_
#define ML_QTL_Type_H_

namespace ML
{

  // Defines a type carrying another type
  template<typename T>
  struct TMLType { using type = T; };

  template<typename T>
  using TMLType_t = typename TMLType<T>::type;

  // Always evaluates to void
  template<typename T>
  struct TMLVoid : TMLType<void> {};
  
  template<typename T>
  using TMLVoid_t = typename TMLVoid<T>::type;

}

#endif