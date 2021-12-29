// Copyright 2021, Philipp Neufeld

#ifndef ML_QTL_Boolean_H_
#define ML_QTL_Boolean_H_

// Includes
#include "Constant.h"

namespace ML
{

  // Type representing a boolean value
  template<bool val>
  using TMLBooleanConstant = TMLConstant<bool, val>;

  // True and False types
  using MLTrueType = TMLBooleanConstant<true>;
  using MLFalseType = TMLBooleanConstant<false>;

  // Always evaluates to true, disregarding the template parameters 
  // This can for example be used in SFINAE expression
  template<typename... Ts>
  struct TMLTrue : MLTrueType {};

  template<typename... Ts>
  constexpr bool TMLTrue_v = TMLTrue<Ts...>::value;

  // Always evaluates to false, disregarding the template parameters 
  // This can for example be used in SFINAE expression
  template<typename... Ts>
  struct TMLFalse : MLFalseType {};

  template<typename... Ts>
  constexpr bool TMLFalse_v = TMLFalse<Ts...>::value;   
  
  // Boolean and for variadic boolean pack
  template<bool v, bool... vs>
  struct TMLBooleanAnd : TMLConstant<bool, v && TMLBooleanAnd<vs...>::value> {};
  template<bool v>
  struct TMLBooleanAnd<v> : TMLConstant<bool, v> {};

  template<bool v, bool... vs>
  constexpr bool TMLBooleanAnd_v = TMLBooleanAnd<v, vs...>::value;  

  // Boolean or for variadic boolean pack
  template<bool v, bool... vs>
  struct TMLBooleanOr : TMLConstant<bool, v || TMLBooleanOr<vs...>::value> {};
  template<bool v>
  struct TMLBooleanOr<v> : TMLConstant<bool, v> {};

  template<bool v, bool... vs>
  constexpr bool TMLBooleanOr_v = TMLBooleanOr<v, vs...>::value;

}

#endif
