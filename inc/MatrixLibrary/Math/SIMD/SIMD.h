// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_SIMD_SIMD_H_
#define ML_MATH_SIMD_SIMD_H_

// Includes
#include <cstdint>
#include <type_traits>
#include <complex>

#include "../MathPrerequisites.h"
#include "../../QTL/Type.h"

#include "../../QTL/EnableIf.h"
#include "../../QTL/Constant.h"
#include "../../QTL/Comparison.h"

namespace ML
{

  template <typename SIMD>
  struct TMLSIMD : public TMLCRTP<SIMD> {};

  // TODO: SIMD Traits


  //
  //  SIMD helper types
  //

  namespace Internal
  {
    // Helper for all non-vectorized intrinsic types
    template <typename ET> 
    struct TMLSIMDIntrinsicsDefaultHelper
    {
      using type = ET;
      template <typename T>
      using EnableIfLoadStore = TMLEnableIf_t<std::is_same<T, ET>::value>;

      // Default value
      QM_ALWAYS_INLINE static type CreateZero() noexcept { return 0; }

      // Load
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadAligned(const T *ptr) noexcept { return *static_cast<const type*>(ptr); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadUnaligned(const T *ptr) noexcept { return *static_cast<const type*>(ptr); }

      // Store
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreAligned(type v, T *ptr) noexcept { *static_cast<type*>(ptr) = v; }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreUnaligned(type v, T *ptr) noexcept { *static_cast<type*>(ptr) = v; }

      // Stream
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        Stream(type v, T *ptr) noexcept { *static_cast<type*>(ptr) = v; }

      // Set1
      QM_ALWAYS_INLINE static type Set1(ET val) noexcept { return val; }
    };

#if defined(ML_MATH_SSE)
    // Helper for types using the SSE __m128 data type
    struct TMLSIMDIntrinsicFloatSSEHelper
    {
      using type = __m128;
      template <typename T>
      using EnableIfLoadStore = TMLEnableIf_t<
        std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value>;

      //Default value
      QM_ALWAYS_INLINE static type CreateZero() noexcept { return _mm_setzero_ps(); }

      // Load
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadAligned(const T *ptr) noexcept { return _mm_load_ps(reinterpret_cast<const float*>(ptr)); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadUnaligned(const T *ptr) noexcept { return _mm_loadu_ps(reinterpret_cast<const float*>(ptr)); }

      // Store
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreAligned(type v, T *ptr) noexcept { _mm_store_ps(reinterpret_cast<float*>(ptr), v); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreUnaligned(type v, T *ptr) noexcept { _mm_storeu_ps(reinterpret_cast<float*>(ptr), v); }

      // Stream
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        Stream(type v, T *ptr) noexcept { _mm_stream_ps(reinterpret_cast<float*>(ptr), v); }

      // Set1
      QM_ALWAYS_INLINE static type Set1(float val) noexcept { return _mm_set1_ps(val); }
      QM_ALWAYS_INLINE static type Set1(std::complex<float> val) noexcept
      {
        const float re = val.real();
        const float im = val.imag();
        return _mm_set_ps(im, re, im, re);
      }
    };
#endif

#if defined(ML_MATH_SSE2)
    // Helper for types using the SSE2 __m128d data type
    struct TMLSIMDIntrinsicDoubleSSE2Helper
    {
      using type = __m128d;
      template <typename T>
      using EnableIfLoadStore = TMLEnableIf_t<
        std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value>;

      //Default value
      QM_ALWAYS_INLINE static type CreateZero() noexcept { return _mm_setzero_pd(); }

      // Load
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadAligned(const T *ptr) noexcept { return _mm_load_pd(reinterpret_cast<const double*>(ptr)); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadUnaligned(const T *ptr) noexcept { return _mm_loadu_pd(reinterpret_cast<const double*>(ptr)); }

      // Store
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreAligned(type v, T *ptr) noexcept { _mm_store_pd(reinterpret_cast<double*>(ptr), v); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreUnaligned(type v, T *ptr) noexcept { _mm_storeu_pd(reinterpret_cast<double*>(ptr), v); }

      // Stream
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        Stream(type v, T *ptr) noexcept { _mm_stream_pd(reinterpret_cast<double*>(ptr), v); }

      // Set1
      QM_ALWAYS_INLINE static type Set1(double val) noexcept { return _mm_set1_pd(val); }
      QM_ALWAYS_INLINE static type Set1(std::complex<double> val) noexcept { return _mm_set_pd(val.imag(), val.real()); }
    };

    // Helper for types using the SSE2 __m128i data type
    struct TMLSIMDIntrinsicIntegerSSE2Helper
    {
      using type = __m128i;
      template <typename T>
      struct EnableIfLoadStore : TMLEnableIf_t<std::is_integral<T>::value> {};
      template <typename T>
      struct EnableIfLoadStore<std::complex<T>> : EnableIfLoadStore<T> {};

      //Default value
      QM_ALWAYS_INLINE static type CreateZero() noexcept { return _mm_setzero_si128(); }

      // Load
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadAligned(const T *ptr) noexcept { return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr)); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadUnaligned(const T *ptr) noexcept { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)); }

      // Store
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreAligned(type v, T *ptr) noexcept { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), v); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreUnaligned(type v, T *ptr) noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v); }

      // Stream
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        Stream(type v, T *ptr) noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), v); }

      // Set1
      QM_ALWAYS_INLINE static type Set1(std::int8_t val) noexcept { return _mm_set1_epi8(val); }
      QM_ALWAYS_INLINE static type Set1(std::uint8_t val) noexcept { return _mm_set1_epi8(*(reinterpret_cast<std::int8_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::int16_t val) noexcept { return _mm_set1_epi16(val); }
      QM_ALWAYS_INLINE static type Set1(std::uint16_t val) noexcept { return _mm_set1_epi16(*(reinterpret_cast<std::int16_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::int32_t val) noexcept { return _mm_set1_epi32(val); }
      QM_ALWAYS_INLINE static type Set1(std::uint32_t val) noexcept { return _mm_set1_epi32(*(reinterpret_cast<std::int32_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::int64_t val) noexcept { return _mm_set1_epi64x(val); }
      QM_ALWAYS_INLINE static type Set1(std::uint64_t val) noexcept { return _mm_set1_epi64x(*(reinterpret_cast<std::int64_t*>(&val))); }

      QM_ALWAYS_INLINE static type Set1(std::complex<std::int8_t> val) noexcept { return _mm_set1_epi16(*(reinterpret_cast<std::int8_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::uint8_t> val) noexcept { return _mm_set1_epi16(*(reinterpret_cast<std::int8_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::int16_t> val) noexcept { return _mm_set1_epi32(*(reinterpret_cast<std::int16_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::uint16_t> val) noexcept { return _mm_set1_epi32(*(reinterpret_cast<std::int16_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::int32_t> val) noexcept { return _mm_set1_epi64x(*(reinterpret_cast<std::int32_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::uint32_t> val) noexcept { return _mm_set1_epi64x(*(reinterpret_cast<std::int32_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::int64_t> val) noexcept
      {
        const std::int64_t re = val.real();
        const std::int64_t im = val.imag();
        return _mm_set_epi64x(im, re);
      }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::uint64_t> val) noexcept
      {
        const std::uint64_t ure = val.real();
        const std::uint64_t uim = val.imag();
        const std::int64_t re = *(reinterpret_cast<const std::int64_t*>(&ure));
        const std::int64_t im = *(reinterpret_cast<const std::int64_t*>(&uim));
        return _mm_set_epi64x(im, re);
      }
    };
#endif

#if defined(ML_MATH_AVX)
    // Helper for types using the AVX __m256 data type
    struct TMLSIMDIntrinsicFloatAVXHelper
    {
      using type = __m256;
      template <typename T>
      using EnableIfLoadStore = TMLEnableIf_t<
        std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value>;

      //Default value
      QM_ALWAYS_INLINE static type CreateZero() noexcept { return _mm256_setzero_ps(); }

      // Load
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadAligned(const T *ptr) noexcept { return _mm256_load_ps(reinterpret_cast<const float*>(ptr)); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadUnaligned(const T *ptr) noexcept { return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr)); }

      // Store
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreAligned(type v, T *ptr) noexcept { _mm256_store_ps(reinterpret_cast<float*>(ptr), v); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreUnaligned(type v, T *ptr) noexcept { _mm256_storeu_ps(reinterpret_cast<float*>(ptr), v); }

      // Stream
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        Stream(type v, T *ptr) noexcept { _mm256_stream_ps(reinterpret_cast<float*>(ptr), v); }
      
      // Set1
      QM_ALWAYS_INLINE static type Set1(float val) noexcept { return _mm256_set1_ps(val); }
      QM_ALWAYS_INLINE static type Set1(std::complex<float> val) noexcept
      {
        const float re = val.real();
        const float im = val.imag();
        return _mm256_set_ps(im, re, im, re, im, re, im, re);
      }
    };

    // Helper for types using the AVX __m256d data type
    struct TMLSIMDIntrinsicDoubleAVXHelper
    {
      using type = __m256d;
      template <typename T>
      using EnableIfLoadStore = TMLEnableIf_t<
        std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value>;

      //Default value
      QM_ALWAYS_INLINE static type CreateZero() noexcept { return _mm256_setzero_pd(); }

      // Load
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadAligned(const T *ptr) noexcept { return _mm256_load_pd(reinterpret_cast<const double*>(ptr)); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadUnaligned(const T *ptr) noexcept { return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr)); }

      // Store
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreAligned(type v, T *ptr) noexcept { _mm256_store_pd(reinterpret_cast<double*>(ptr), v); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreUnaligned(type v, T *ptr) noexcept { _mm256_storeu_pd(reinterpret_cast<double*>(ptr), v); }

      // Stream
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        Stream(type v, T *ptr) noexcept { _mm256_stream_pd(reinterpret_cast<double*>(ptr), v); }
      
      // Set1
      QM_ALWAYS_INLINE static type Set1(double val) noexcept { return _mm256_set1_pd(val); }
      QM_ALWAYS_INLINE static type Set1(std::complex<double> val) noexcept
      {
        const double re = val.real();
        const double im = val.imag();
        return _mm256_set_pd(im, re, im, re);
      }
    };
#endif

#if defined(ML_MATH_AVX2)
    // Helper for types using the AVX2 __m256i data type
    struct TMLSIMDIntrinsicIntegerAVX2Helper
    {
      using type = __m256i;
      template <typename T>
      struct EnableIfLoadStore : TMLEnableIf_t<std::is_integral<T>::value> {};
      template <typename T>
      struct EnableIfLoadStore<std::complex<T>> : EnableIfLoadStore<T> {};

      template <typename T, std::size_t size>
      using EnableIfSet1 = TMLEnableIf_t<std::is_integral<T>::value && (sizeof(T) == size)>;

      //Default value
      QM_ALWAYS_INLINE static type CreateZero() noexcept { return _mm256_setzero_si256(); }

      // Load
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadAligned(const T *ptr) noexcept { return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr)); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static type
        LoadUnaligned(const T *ptr) noexcept { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)); }

      // Store
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreAligned(type v, T *ptr) noexcept { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), v); }
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        StoreUnaligned(type v, T *ptr) noexcept { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), v); }

      // Stream
      template <typename T, typename = EnableIfLoadStore<T>> QM_ALWAYS_INLINE static void
        Stream(type v, T *ptr) noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), v); }

      // Set1
      QM_ALWAYS_INLINE static type Set1(std::int8_t val) noexcept { return _mm256_set1_epi8(val); }
      QM_ALWAYS_INLINE static type Set1(std::uint8_t val) noexcept { return _mm256_set1_epi8(*(reinterpret_cast<std::int8_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::int16_t val) noexcept { return _mm256_set1_epi16(val); }
      QM_ALWAYS_INLINE static type Set1(std::uint16_t val) noexcept { return _mm256_set1_epi16(*(reinterpret_cast<std::int16_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::int32_t val) noexcept { return _mm256_set1_epi32(val); }
      QM_ALWAYS_INLINE static type Set1(std::uint32_t val) noexcept { return _mm256_set1_epi32(*(reinterpret_cast<std::int32_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::int64_t val) noexcept { return _mm256_set1_epi64x(val); }
      QM_ALWAYS_INLINE static type Set1(std::uint64_t val) noexcept { return _mm256_set1_epi64x(*(reinterpret_cast<std::int64_t*>(&val))); }

      QM_ALWAYS_INLINE static type Set1(std::complex<std::int8_t> val) noexcept { return _mm256_set1_epi16(*(reinterpret_cast<std::int8_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::uint8_t> val) noexcept { return _mm256_set1_epi16(*(reinterpret_cast<std::int8_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::int16_t> val) noexcept { return _mm256_set1_epi32(*(reinterpret_cast<std::int16_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::uint16_t> val) noexcept { return _mm256_set1_epi32(*(reinterpret_cast<std::int16_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::int32_t> val) noexcept { return _mm256_set1_epi64x(*(reinterpret_cast<std::int32_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::uint32_t> val) noexcept { return _mm256_set1_epi64x(*(reinterpret_cast<std::int32_t*>(&val))); }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::int64_t> val) noexcept
      {
        const std::int64_t re = val.real();
        const std::int64_t im = val.imag();
        return _mm256_set_epi64x(im, re, im, re);
      }
      QM_ALWAYS_INLINE static type Set1(std::complex<std::uint64_t> val) noexcept
      {
        const std::uint64_t ure = val.real();
        const std::uint64_t uim = val.imag();
        const std::int64_t re = *(reinterpret_cast<const std::int64_t*>(&ure));
        const std::int64_t im = *(reinterpret_cast<const std::int64_t*>(&uim));
        return _mm256_set_epi64x(im, re, im, re);
      }
    };
#endif

    // Implementation of the actual intrinsic type used in the math library
    template <template <typename> class CRTP, typename ET, typename IntrinsicsHelper, typename=void> 
    struct TMLSIMD_impl;
    template <template <typename> class CRTP, typename ET, typename IntrinsicsHelper> 
    struct TMLSIMD_impl<CRTP, ET, IntrinsicsHelper, TMLEnableIf_t<sizeof(typename IntrinsicsHelper::type) % sizeof(ET) == 0>>
      : CRTP<TMLSIMD_impl<CRTP, ET, IntrinsicsHelper>>
    {
      using ElementType = ET;
      using IntrinsicType = typename IntrinsicsHelper::type;
      constexpr static std::size_t Size_v = sizeof(IntrinsicType) / sizeof(ElementType);

      QM_ALWAYS_INLINE TMLSIMD_impl() noexcept : m_value(IntrinsicsHelper::CreateZero()) {}
      QM_ALWAYS_INLINE TMLSIMD_impl(IntrinsicType val) noexcept : m_value(val) {}
      QM_ALWAYS_INLINE TMLSIMD_impl(const TMLSIMD_impl &rhs) noexcept : m_value(rhs.m_value) {}
      QM_ALWAYS_INLINE ElementType operator[](std::size_t i) const noexcept { return (reinterpret_cast<const ElementType*>(&m_value))[i]; }
      QM_ALWAYS_INLINE ElementType &operator[](std::size_t i) noexcept { return (reinterpret_cast<ElementType*>(&m_value))[i]; }

      template <typename T> QM_ALWAYS_INLINE static TMLSIMD_impl 
        LoadAligned(const T *ptr) noexcept { return IntrinsicsHelper::LoadAligned(ptr); }
      template <typename T> QM_ALWAYS_INLINE static TMLSIMD_impl 
        LoadUnaligned(const T *ptr) noexcept { return IntrinsicsHelper::LoadUnaligned(ptr); }

      template <typename T> QM_ALWAYS_INLINE static void 
        StoreAligned(TMLSIMD_impl v, T *ptr) noexcept { IntrinsicsHelper::StoreAligned(v.m_value, ptr); }
      template <typename T> QM_ALWAYS_INLINE static void 
        StoreUnaligned(TMLSIMD_impl v, T *ptr) noexcept { IntrinsicsHelper::StoreUnaligned(v.m_value, ptr); }
      template <typename T> QM_ALWAYS_INLINE static void 
        Stream(TMLSIMD_impl v, T *ptr) noexcept { IntrinsicsHelper::Stream(v.m_value, ptr); }

      QM_ALWAYS_INLINE static TMLSIMD_impl 
        SetZero() noexcept { return IntrinsicsHelper::CreateZero(); }
      template <typename T> QM_ALWAYS_INLINE static TMLSIMD_impl 
        Set1(T val) noexcept { return IntrinsicsHelper::Set1(val); }

      IntrinsicType m_value;
    };
  }

  //
  // Traits
  //

  // Trait that checks if type is a SIMD type
  template<typename T>
  struct TMLIsSIMD : TMLIsCRTP<T, TMLSIMD> {};
  
  template<typename T>
  constexpr bool TMLIsSIMD_v = TMLIsSIMD<T>::value;

  // Get the inner CRTP type
  template<typename T, typename=void>
  struct TMLDecaySIMDType;
  template<typename T>
  struct TMLDecaySIMDType<T, TMLEnableIf_t<TMLIsSIMD_v<T>>> 
    : TMLDecayCRTP<T> {};
  
  template<typename T>
  using TMLDecaySIMDType_t = typename TMLDecaySIMDType<T>::type;

  // Get element type of a SIMD type
  template<typename T, typename=void>
  struct TMLSIMDElementType;
  template<typename T>
  struct TMLSIMDElementType<T, TMLEnableIf_t<TMLIsSIMD_v<T> && TMLTrue_v<typename TMLDecaySIMDType_t<T>::ElementType>>>
    : TMLType<typename TMLDecaySIMDType_t<T>::ElementType> {};

  template<typename T>
  using TMLSIMDElementType_t = typename TMLSIMDElementType<T>::type;

  // Get SIMD size of a SIMD type
  template<typename T, typename=void>
  struct TMLSIMDSize;
  template<typename T>
  struct TMLSIMDSize<T, TMLEnableIf_t<TMLIsSIMD_v<T>>>
    : TMLConstant<std::size_t, TMLDecaySIMDType_t<T>::Size_v> {};

  template<typename T>
  constexpr auto TMLSIMDSize_v = TMLSIMDSize<T>::value;


  //
  // Definition of SIMD types
  //

  // Non-vectorized SIMD types
  template<typename T>
  using TMLSIMDDefault = Internal::TMLSIMD_impl<
    TMLSIMD, 
    T, 
    Internal::TMLSIMDIntrinsicsDefaultHelper<T>>;

#if defined(ML_MATH_SSE)
  // SSE CRTP types
  template <typename SIMD> struct TMLSIMDSSE : TMLSIMD<SIMD> {};
  template <typename SIMD> struct TMLSIMD32floatSSE : TMLSIMDSSE<SIMD> {};

  // SSE SIMD types
  using MLSIMD32fSSE = Internal::TMLSIMD_impl<
    TMLSIMD32floatSSE, 
    float, 
    Internal::TMLSIMDIntrinsicFloatSSEHelper>;

  using MLSIMD32cfSSE = Internal::TMLSIMD_impl<
    TMLSIMD32floatSSE, 
    std::complex<float>, 
    Internal::TMLSIMDIntrinsicFloatSSEHelper>;

#endif

#if defined(ML_MATH_SSE2)
  // SSE2 CRTP types
  template <typename SIMD> struct TMLSIMDSSE2 : TMLSIMD<SIMD> {};
  template <typename SIMD> struct TMLSIMDintSSE2 : TMLSIMDSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD8intSSE2 : TMLSIMDintSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD16intSSE2 : TMLSIMDintSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD32intSSE2 : TMLSIMDintSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD64intSSE2 : TMLSIMDintSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD8uiSSE2 : TMLSIMD8intSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD16uiSSE2 : TMLSIMD16intSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD32uiSSE2 : TMLSIMD32intSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD64uiSSE2 : TMLSIMD64intSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD8cuiSSE2 : TMLSIMD8intSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD16cuiSSE2 : TMLSIMD16intSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD32cuiSSE2 : TMLSIMD32intSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD64cuiSSE2 : TMLSIMD64intSSE2<SIMD> {};
  template <typename SIMD> struct TMLSIMD64floatSSE2 : TMLSIMDSSE2<SIMD> {};

  // SSE2 SIMD types
  using MLSIMD64fSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD64floatSSE2, 
    double, 
    Internal::TMLSIMDIntrinsicDoubleSSE2Helper>;
  using MLSIMD8uSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD8uiSSE2, 
    std::uint8_t, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD8iSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD8uiSSE2, 
    std::int8_t, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD16uSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD16uiSSE2, 
    std::uint16_t, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD16iSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD16uiSSE2, 
    std::int16_t, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD32uSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD32uiSSE2, 
    std::uint32_t, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD32iSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD32uiSSE2, 
    std::int32_t, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD64uSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD64uiSSE2, 
    std::uint64_t, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD64iSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD64uiSSE2, 
    std::int64_t, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD64cfSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD64floatSSE2, 
    std::complex<double>, 
    Internal::TMLSIMDIntrinsicDoubleSSE2Helper>;

  using MLSIMD8cuSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD8cuiSSE2, 
    std::complex<std::uint8_t>, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD8ciSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD8cuiSSE2, 
    std::complex<std::int8_t>, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD16cuSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD16cuiSSE2, 
    std::complex<std::uint16_t>, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD16ciSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD16cuiSSE2, 
    std::complex<std::int16_t>, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD32cuSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD32cuiSSE2, 
    std::complex<std::uint32_t>, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD32ciSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD32cuiSSE2, 
    std::complex<std::int32_t>, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD64cuSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD64cuiSSE2, 
    std::complex<std::uint64_t>, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

  using MLSIMD64ciSSE2 = Internal::TMLSIMD_impl<
    TMLSIMD64cuiSSE2, 
    std::complex<std::int64_t>, 
    Internal::TMLSIMDIntrinsicIntegerSSE2Helper>;

#endif

#if defined(ML_MATH_AVX)
  // SSE CRTP types
  template <typename SIMD> struct TMLSIMDAVX : TMLSIMD<SIMD> {};
  template <typename SIMD> struct TMLSIMD32floatAVX : TMLSIMDAVX<SIMD> {};
  template <typename SIMD> struct TMLSIMD64floatAVX : TMLSIMDAVX<SIMD> {};

  // AVX SIMD types
  using MLSIMD32fAVX = Internal::TMLSIMD_impl<
    TMLSIMD32floatAVX, 
    float, 
    Internal::TMLSIMDIntrinsicFloatAVXHelper>;

  using MLSIMD64fAVX = Internal::TMLSIMD_impl<
    TMLSIMD64floatAVX, 
    double, 
    Internal::TMLSIMDIntrinsicDoubleAVXHelper>;

  using MLSIMD32cfAVX = Internal::TMLSIMD_impl<
    TMLSIMD32floatAVX, 
    std::complex<float>, 
    Internal::TMLSIMDIntrinsicFloatAVXHelper>;

  using MLSIMD64cfAVX = Internal::TMLSIMD_impl<
    TMLSIMD64floatAVX, 
    std::complex<double>, 
    Internal::TMLSIMDIntrinsicDoubleAVXHelper>;

#endif

#if defined(ML_MATH_AVX2)
  // AVX2 CRTP types
  template <typename SIMD> struct TMLSIMDAVX2 : TMLSIMD<SIMD> {};
  template <typename SIMD> struct TMLSIMDintAVX2 : TMLSIMDAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD8intAVX2 : TMLSIMDintAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD16intAVX2 : TMLSIMDintAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD32intAVX2 : TMLSIMDintAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD64intAVX2 : TMLSIMDintAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD8uiAVX2 : TMLSIMD8intAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD16uiAVX2 : TMLSIMD16intAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD32uiAVX2 : TMLSIMD32intAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD64uiAVX2 : TMLSIMD64intAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD8cuiAVX2 : TMLSIMD8intAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD16cuiAVX2 : TMLSIMD16intAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD32cuiAVX2 : TMLSIMD32intAVX2<SIMD> {};
  template <typename SIMD> struct TMLSIMD64cuiAVX2 : TMLSIMD64intAVX2<SIMD> {};


  // AVX2 SIMD types
  using MLSIMD8uAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD8uiAVX2, 
    std::uint8_t, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD8iAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD8uiAVX2, 
    std::int8_t, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD16uAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD16uiAVX2, 
    std::uint16_t, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD16iAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD16uiAVX2, 
    std::int16_t, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD32uAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD32uiAVX2, 
    std::uint32_t, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD32iAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD32uiAVX2, 
    std::int32_t, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD64uAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD64uiAVX2, 
    std::uint64_t, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD64iAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD64uiAVX2, 
    std::int64_t, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD8cuAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD8cuiAVX2, 
    std::complex<std::uint8_t>, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD8ciAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD8cuiAVX2, 
    std::complex<std::int8_t>, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD16cuAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD16cuiAVX2, 
    std::complex<std::uint16_t>, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD16ciAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD16cuiAVX2, 
    std::complex<std::int16_t>, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD32cuAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD32cuiAVX2, 
    std::complex<std::uint32_t>, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD32ciAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD32cuiAVX2, 
    std::complex<std::int32_t>, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD64cuAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD64cuiAVX2, 
    std::complex<std::uint64_t>, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

  using MLSIMD64ciAVX2 = Internal::TMLSIMD_impl<
    TMLSIMD64cuiAVX2, 
    std::complex<std::int64_t>, 
    Internal::TMLSIMDIntrinsicIntegerAVX2Helper>;

#endif

  //
  // Selection of a SIMD type based on the base type
  //
  
  namespace Internal
  {
    template<typename T, typename TDes, std::size_t MaxSize, std::size_t Size>
    using TMLSIMDSelEnableIf_t = TMLEnableIf_t<TMLBooleanAnd_v<
      (Size & (Size-1)) == 0, // is power of 2
      std::is_same<T, TDes>::value, 
      TMLGreaterEqual_v<std::size_t, MaxSize, Size>, 
      TMLLess_v<std::size_t, MaxSize, 2*Size>>>;
  }

  template<typename T, std::size_t maxSize, typename=void>
  struct TMLSIMDTypeSelector : TMLSIMDTypeSelector<T, maxSize / 2> {};

  // Disable invalid combination
  template<typename T>
  struct TMLSIMDTypeSelector<T, 0, void>;
  template<std::size_t maxSize>
  struct TMLSIMDTypeSelector<void, maxSize, void>;
  template<>
  struct TMLSIMDTypeSelector<void, 0, void>;

  // Non-vectorized types
  template<typename T>
  struct TMLSIMDTypeSelector<T, 1, void> : TMLType<TMLSIMDDefault<T>> {};

#if defined(ML_MATH_SSE)
  // SSE single precision floating point types
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, float, maxSize, 4>>
    : TMLType<MLSIMD32fSSE> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<float>, maxSize, 2>>
    : TMLType<MLSIMD32cfSSE> {};
#endif

#if defined(ML_MATH_SSE2)
  // SSE2 double precision floating point types
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, double, maxSize, 2>>
    : TMLType<MLSIMD64fSSE2> {};
  template<>
  struct TMLSIMDTypeSelector<std::complex<double>, 1, void>
    : TMLType<MLSIMD64cfSSE2> {};

  // SSE2 integer types
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::int8_t, maxSize, 16>>
    : TMLType<MLSIMD8iSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::uint8_t, maxSize, 16>>
    : TMLType<MLSIMD8uSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::int16_t, maxSize, 8>>
    : TMLType<MLSIMD16iSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::uint16_t, maxSize, 8>>
    : TMLType<MLSIMD16uSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::int32_t, maxSize, 4>>
    : TMLType<MLSIMD32iSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::uint32_t, maxSize, 4>>
    : TMLType<MLSIMD32uSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::int64_t, maxSize, 2>>
    : TMLType<MLSIMD64iSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::uint64_t, maxSize, 2>>
    : TMLType<MLSIMD64uSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::int8_t>, maxSize, 8>>
    : TMLType<MLSIMD8ciSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::uint8_t>, maxSize, 8>>
    : TMLType<MLSIMD8cuSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::int16_t>, maxSize, 4>>
    : TMLType<MLSIMD16ciSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::uint16_t>, maxSize, 4>>
    : TMLType<MLSIMD16cuSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::int32_t>, maxSize, 2>>
    : TMLType<MLSIMD32ciSSE2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::uint32_t>, maxSize, 2>>
    : TMLType<MLSIMD32cuSSE2> {};
  template<>
  struct TMLSIMDTypeSelector<std::complex<std::int64_t>, 1, void>
    : TMLType<MLSIMD64ciSSE2> {};
  template<>
  struct TMLSIMDTypeSelector<std::complex<std::uint64_t>, 1, void>
    : TMLType<MLSIMD64cuSSE2> {};   
#endif

#if defined(ML_MATH_AVX)
  // AVX single precision floating point types
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, float, maxSize, 8>>
    : TMLType<MLSIMD32fAVX> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<float>, maxSize, 4>>
    : TMLType<MLSIMD32cfAVX> {};

  // AVX double precision floating point types
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, double, maxSize, 4>>
    : TMLType<MLSIMD64fAVX> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<double>, maxSize, 2>>
    : TMLType<MLSIMD64cfAVX> {};
#endif

#if defined(ML_MATH_AVX2)
  // AVX2 integer types
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::int8_t, maxSize, 32>>
    : TMLType<MLSIMD8iAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::uint8_t, maxSize, 32>>
    : TMLType<MLSIMD8uAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::int16_t, maxSize, 16>>
    : TMLType<MLSIMD16iAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::uint16_t, maxSize, 16>>
    : TMLType<MLSIMD16uAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::int32_t, maxSize, 8>>
    : TMLType<MLSIMD32iAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::uint32_t, maxSize, 8>>
    : TMLType<MLSIMD32uAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::int64_t, maxSize, 4>>
    : TMLType<MLSIMD64iAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::uint64_t, maxSize, 4>>
    : TMLType<MLSIMD64uAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::int8_t>, maxSize, 16>>
    : TMLType<MLSIMD8ciAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::uint8_t>, maxSize, 16>>
    : TMLType<MLSIMD8cuAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::int16_t>, maxSize, 8>>
    : TMLType<MLSIMD16ciAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::uint16_t>, maxSize, 8>>
    : TMLType<MLSIMD16cuAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::int32_t>, maxSize, 4>>
    : TMLType<MLSIMD32ciAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::uint32_t>, maxSize, 4>>
    : TMLType<MLSIMD32cuAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::int64_t>, maxSize, 2>>
    : TMLType<MLSIMD64ciAVX2> {};
  template<typename T, std::size_t maxSize>
  struct TMLSIMDTypeSelector<T, maxSize, Internal::TMLSIMDSelEnableIf_t<T, std::complex<std::uint64_t>, maxSize, 2>>
    : TMLType<MLSIMD64cuAVX2> {};
#endif

  template<typename T, std::size_t maxSize>
  using TMLSIMDTypeSelector_t = typename TMLSIMDTypeSelector<T, maxSize>::type;

}

#include "Add.h"
#include "Sub.h"
#include "Mul.h"
#include "Div.h"
#include "FMA.h"
#include "Broadcast.h"

#endif
