// Copyright 2021, Philipp Neufeld

// The ML Math library is inspired by the Blaze library by Klaus Iglberger

#ifndef ML_MATH_MathPrerequisites_H_
#define ML_MATH_MathPrerequisites_H_

// Includes
#include <cstdint>
#include <cmath>

#include "../Core/Platform.h"
#include "../QTL/Type.h"
#include "../QTL/EnableIf.h"
#include "../QTL/Constant.h"
#include "../QTL/Comparison.h"

/*
*	NOTICE:
*	Do NOT use C++ unions in performance critical code because they force 
* variables to be stored in memory even if they could be stored in a register. 
* see Agner Fog's optimization manual for further information.
*/

// define ML_MATH_NO_INTRINSICS for the native C++ implementation.

// Include appropriate intrinsics header
#if !defined(ML_MATH_NO_INTRINSICS)
#include <immintrin.h>
#endif

// Check for Features -> macros are defined by the accompanying CMake script
#if !defined(ML_MATH_NO_INTRINSICS)
#if defined(ML_FMA) && !defined(ML_MATH_FMA)
#define ML_MATH_FMA
#endif
#if defined(ML_AVX2) && !defined(ML_MATH_AVX2)
#define ML_MATH_AVX2
#endif
#if defined(ML_AVX) && !defined(ML_MATH_AVX)
#define ML_MATH_AVX
#endif
#if defined(ML_SSE4_2) && !defined(ML_MATH_SSE4_2)
#define ML_MATH_SSE4_2
#endif
#if defined(ML_SSE4_1) && !defined(ML_MATH_SSE4_1)
#define ML_MATH_SSE4_1
#endif
#if defined(ML_SSSE3) && !defined(ML_MATH_SSSE3)
#define ML_MATH_SSSE3
#endif
#if defined(ML_SSE3) && !defined(ML_MATH_SSE3)
#define ML_MATH_SSE3
#endif
#if defined(ML_SSE2) && !defined(ML_MATH_SSE2)
#define ML_MATH_SSE2
#endif
#if defined(ML_SSE) && !defined(ML_MATH_SSE)
#define ML_MATH_SSE
#endif
#endif


// Alignment
#if !defined(ML_MATH_NO_INTRINSICS) && !defined(ML_MATH_ALIGNAS32)
#define ML_MATH_ALIGNAS32 alignas(32)
#else
#define ML_MATH_ALIGNAS
#endif

#if !defined(ML_MATH_NO_INTRINSICS) && !defined(ML_MATH_ALIGNAS16)
#define ML_MATH_ALIGNAS16 alignas(32)
#else
#define ML_MATH_ALIGNAS16
#endif

// Calling convention
#if !defined(ML_MATH_NO_INTRINSICS)
#if defined(ML_COMPILER_MSVC)
#if _MSC_FULL_VER >= 180020418
#define ML_MATH_VECTORCALL
#define QM_CALLCONV __vectorcall
#else
#define ML_MATH_FASTCALL
#define QM_CALLCONV __fastcall
#endif
#elif defined(ML_COMPILER_CLANG)
#if __clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 6)
#define ML_MATH_VECTORCALL
#define QM_CALLCONV __vectorcall
#else
#define ML_MATH_FASTCALL
#define QM_CALLCONV __fastcall
#endif
#endif
#endif

#if !defined(QM_CALLCONV)
#define QM_CALLCONV
#endif

// Inline
#if defined(ML_COMPILER_MSVC)
#define QM_ALWAYS_INLINE __forceinline
#elif defined(ML_COMPILER_GNUC) || defined(ML_COMPILER_CLANG)
#define QM_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif

#ifndef QM_CONST
#define QM_CONST constexpr
#endif // !QM_CONST

namespace ML
{

  // essential scalar constants
  QM_CONST double QM_PI = 3.141592653589793238462643;
  QM_CONST double QM_2PI = 2.0 * QM_PI;
  QM_CONST double QM_1DIVPI = 1.0 / QM_PI;
  QM_CONST double QM_1DIV2PI = 1.0 / QM_2PI;
  QM_CONST double QM_PIDIV2 = QM_PI / 2.0;
  QM_CONST double QM_PIDIV4 = QM_PI / 4.0;
  QM_CONST double QM_E = 2.718281828459045235360;
  QM_CONST double QM_GOLDENRATIO = 1.618033988749894848204586;

  QM_CONST float QM_F_PI = static_cast<float>(QM_PI);
  QM_CONST float QM_F_2PI = static_cast<float>(QM_2PI);
  QM_CONST float QM_F_1DIVPI = static_cast<float>(QM_1DIVPI);
  QM_CONST float QM_F_1DIV2PI = static_cast<float>(QM_1DIV2PI);
  QM_CONST float QM_F_PIDIV2 = static_cast<float>(QM_PIDIV2);
  QM_CONST float QM_F_PIDIV4 = static_cast<float>(QM_PIDIV4);
  QM_CONST float QM_F_E = static_cast<float>(QM_E);
  QM_CONST float QM_F_GOLDENRATIO = static_cast<float>(QM_GOLDENRATIO);

  // degree <-> radian
  inline constexpr float ConvertToRadians(float degrees) { return degrees * (QM_F_PI / 180.0f); }
  inline constexpr double ConvertToRadians(double degrees) { return degrees * (QM_PI / 180.0); }
  inline constexpr float ConvertToDegrees(float radians) { return radians * (180.0f / QM_F_PI); }
  inline constexpr double ConvertToDegrees(double radians) { return radians * (180.0 / QM_PI); }


  // CRTP
  template <typename T>
  struct TMLCRTP
  {
    // CRTP conversion operators to the actual vector type (type-safe downcast)
    QM_ALWAYS_INLINE constexpr T& operator~() noexcept { return static_cast<T&>(*this); }
    QM_ALWAYS_INLINE constexpr const T& operator~() const noexcept { return static_cast<const T&>(*this); }
  };

  
  // Trait that checks if type is a crtp type
  template<typename T, template<typename> class CRTP=TMLCRTP, typename=void>
  struct TMLIsCRTP : std::false_type {};
  template<typename T, template<typename> class CRTP>
  struct TMLIsCRTP<T, CRTP, TMLEnableIf_t<
      std::is_base_of<TMLCRTP<std::decay_t<decltype(~std::declval<T>())>>, std::decay_t<T>>::value &&
      std::is_base_of<CRTP<std::decay_t<decltype(~std::declval<T>())>>, std::decay_t<T>>::value
    >> : std::true_type {};

  template<typename MT>
  constexpr bool TMLIsCRTP_v = TMLIsCRTP<MT>::value;

  // Get the inner CRTP type
  template<typename T, typename=void>
  struct TMLDecayCRTP;
  template<typename T>
  struct TMLDecayCRTP<T, TMLEnableIf_t<TMLIsCRTP_v<T>>>
    : TMLType<std::decay_t<decltype(~std::declval<T>())>> {};

  template<typename T>
  using TMLDecayCRTP_t = typename TMLDecayCRTP<T>::type;

  // Get the inner CRTP type (also works for non-CRTP types and uses std::decay on them)
  template<typename T, typename=void>
  struct TMLDecayCRTPGeneral : TMLType<std::decay_t<T>> {};
  template<typename T>
  struct TMLDecayCRTPGeneral<T, TMLEnableIf_t<TMLIsCRTP_v<T>>>
    : TMLType<std::decay_t<decltype(~std::declval<T>())>> {};

  template<typename T>
  using TMLDecayCRTPGeneral_t = typename TMLDecayCRTPGeneral<T>::type;

  // constexpr for
  // see https://artificial-mind.net/blog/2020/10/31/constexpr-for
  namespace Internal
  {
    template<typename ItType, ItType Start, ItType End, ItType Inc, class F, bool end>
    struct TMLConstexprFor_Helper
    {
      QM_ALWAYS_INLINE constexpr static void Exec(F&& f)
      {
        f(TMLConstant<ItType, Start>());
        TMLConstexprFor_Helper<ItType, Start + Inc, End, Inc, F, TMLLess_v<ItType, Start + Inc, End>>::Exec(std::forward<F>(f));
      }
    };

    template<typename ItType, ItType Start, ItType End, ItType Inc, class F>
    struct TMLConstexprFor_Helper<ItType, Start, End, Inc, F, false>
    {
      QM_ALWAYS_INLINE constexpr static void Exec(F&& f) { }
    };

  }

  template<typename ItType, ItType Start, ItType End, ItType Inc, class F>
  QM_ALWAYS_INLINE constexpr void MLConstexprFor(F&& f)
  {
    Internal::TMLConstexprFor_Helper<ItType, Start, End, Inc, F, TMLLess_v<ItType, Start, End>>::Exec(std::forward<F>(f));
  }

}

#endif
