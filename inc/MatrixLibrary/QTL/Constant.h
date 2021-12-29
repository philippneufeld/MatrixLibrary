// Copyright 2021, Philipp Neufeld

#ifndef ML_QTL_Constant_H_
#define ML_QTL_Constant_H_

namespace ML
{

  // Defines a type representing a value
  template<typename T, T val>
  struct TMLConstant 
  { 
    constexpr static T value = val;
    constexpr operator T() const noexcept { return value; } 
    constexpr T operator()() const noexcept { return value; }
  };
  
}

#endif