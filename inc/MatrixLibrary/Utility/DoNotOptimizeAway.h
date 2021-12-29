// Copyright 2021, Philipp Neufeld

#ifndef ML_DoNotOptimizeAway_H_
#define ML_DoNotOptimizeAway_H_

// Includes
#include "../QTL/EnableIf.h"

namespace ML
{
  // Gets the (real) address of an object, even if the addressof-operator(&) 
  // is overloaded
  template<typename T>
  inline T* MLGetAddressOf(T& obj)
  {
    return reinterpret_cast<T*>(&(const_cast<char&>(
      reinterpret_cast<const volatile char&>(obj))));
  }

  //DoNotOptimizeAway
  // This is mostly taken from the Facebook folly benchmark API
  // if there is a variable c that should not be optimized away by the compiler,
  // use DoNotOptimizeAway(c)
#ifdef ML_COMPILER_MSVC
  namespace Internal
  {
    //function has absolutely no overhead
#pragma optimize("", off)
    template<typename T>
    inline void MLDoNotOptimizeDependencySink(const void*) {}
#pragma optimize("", on)
  }

  template<typename T>
  void MLDoNotOptimizeAway(const T& datum)
  {
    Internal::MLDoNotOptimizeDependencySink<T>(MLGetAddressOf(datum));
  }

  template<typename T>
  void MLMakeUnpredictable(const T& datum)
  {
    Internal::MLDoNotOptimizeDependencySink<T>(MLGetAddressOf(datum));
  }

#elif defined(ML_COMPILER_GNU) || defined(ML_COMPILER_CLANG)

  namespace Internal
  {
    template <typename T>
    struct TMLDoNotOptimizeAwayNeedsIndirect {
      using Decayed = typename std::decay<T>::type;

      // First two constraints ensure it can be an "r" operand.
      // std::is_pointer check is because callers seem to expect that
      // doNotOptimizeAway(&x) is equivalent to doNotOptimizeAway(x).
      constexpr static bool value =
        !std::is_trivially_copyable<Decayed>::value ||
        sizeof(Decayed) > sizeof(long) || std::is_pointer<Decayed>::value;
    };
  }

  template <typename T>
  typename TMLEnableIf<
    !Internal::TMLDoNotOptimizeAwayNeedsIndirect<T>::value
  >::type MLDoNotOptimizeAway(const T& datum) { asm volatile("" ::"r"(datum)); }

  template <typename T>
  typename TMLEnableIf<
    Internal::TMLDoNotOptimizeAwayNeedsIndirect<T>::value
  >::type MLDoNotOptimizeAway(const T& datum)
  {
    asm volatile("" ::"m"(datum) : "memory");
  }

  template <typename T>
  typename TMLEnableIf<
    !Internal::TMLDoNotOptimizeAwayNeedsIndirect<T>::value
  >::type MLMakeUnpredictable(T& datum)
  {
    asm volatile("" : "+r"(datum));
  }

  template <typename T>
  typename TMLEnableIf<
    Internal::TMLDoNotOptimizeAwayNeedsIndirect<T>::value
  >::type MLMakeUnpredictable(T& datum)
  {
    asm volatile("" ::"m"(datum) : "memory");
  }

#else
  template<typename T>
  void MLDoNotOptimizeAway(const T& datum) { }

  template<typename T>
  void MLMakeUnpredictable(const T& datum) { MLDoNotOptimizeAway(datum); }

#endif
}

#endif // !ML_DoNotOptimizeAway_H_
