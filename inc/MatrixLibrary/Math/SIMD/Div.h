// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_SIMD_Div_H_
#define ML_MATH_SIMD_Div_H_

// Includes
#include "SIMD.h"

namespace ML
{

  namespace Internal
  {
    template<typename SIMD>
    class TMLSIMDDefaultDivHelper
    {
    public:
      QM_ALWAYS_INLINE TMLSIMDDefaultDivHelper(TMLDecaySIMDType_t<SIMD>& res, const TMLSIMD<SIMD>& a, const TMLSIMD<SIMD>& b)
        : m_res(res), m_a(a), m_b(b) {}
      QM_ALWAYS_INLINE void operator()(std::size_t i) { m_res[i] = (~m_a)[i] / (~m_b)[i]; }
      
    private:
      TMLDecaySIMDType_t<SIMD>& m_res;
      const TMLSIMD<SIMD>& m_a;
      const TMLSIMD<SIMD>& m_b;
    };
  }

  // default division
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator/(const TMLSIMD<SIMD>& a, const TMLSIMD<SIMD>& b)
  {
    TMLDecaySIMDType_t<SIMD> res;
    Internal::TMLSIMDDefaultDivHelper<SIMD> helper(res, a, b);
    MLConstexprFor<std::size_t, 0, TMLSIMDSize_v<SIMD>, 1>(helper);
    return res;
  }

#if defined(ML_MATH_SSE)
  // SSE 32 bit floating point divion
  QM_ALWAYS_INLINE MLSIMD32fSSE operator/(const MLSIMD32fSSE& a, const MLSIMD32fSSE& b)
  {
    return _mm_div_ps(a.m_value, b.m_value);
  }

  // SSE 32 bit complex floating point division
  QM_ALWAYS_INLINE MLSIMD32cfSSE operator/(const MLSIMD32cfSSE& a, const MLSIMD32cfSSE& b)
  {
    __m128 mask = _mm_set_ps(-0.0f, 0.0f, -0.0f, 0.0f);
    MLSIMD32cfSSE bconj = _mm_xor_ps(b.m_value, mask);
    MLSIMD32cfSSE num = a * bconj;

    __m128 b2 = _mm_mul_ps(b.m_value, b.m_value);
    __m128 b2s = _mm_shuffle_ps(b2, b2, _MM_SHUFFLE(2,3,0,1));
    __m128 den = _mm_add_ps(b2, b2s);

    return _mm_div_ps(num.m_value, den);
  }
#endif

#if defined(ML_MATH_SSE2)
  // SSE2 64 bit floating point division
  QM_ALWAYS_INLINE MLSIMD64fSSE2 operator/(const MLSIMD64fSSE2& a, const MLSIMD64fSSE2& b)
  {
    return _mm_div_pd(a.m_value, b.m_value);
  }

  // SSE2 64 bit floating point division
  QM_ALWAYS_INLINE MLSIMD64cfSSE2 operator/(const MLSIMD64cfSSE2& a, const MLSIMD64cfSSE2& b)
  {
    __m128d mask = _mm_set_pd(-0.0f, 0.0f);
    MLSIMD64cfSSE2 bconj = _mm_xor_pd(b.m_value, mask);
    MLSIMD64cfSSE2 num = a * bconj;

    __m128d b2 = _mm_mul_pd(b.m_value, b.m_value);
    __m128d b2s = _mm_shuffle_pd(b2, b2, _MM_SHUFFLE2(0,1));
    __m128d den = _mm_add_pd(b2, b2s);

    return _mm_div_pd(num.m_value, den);
  }

  // SSE integer division (no vectorized version available)
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator/(const TMLSIMDintSSE2<SIMD>& a, const TMLSIMDintSSE2<SIMD>& b)
  {
    return (~a).m_value / (~b).m_value;
  }
#endif

#if defined(ML_MATH_AVX)
  // AVX 32 bit floating point divsion
  QM_ALWAYS_INLINE MLSIMD32fAVX operator/(const MLSIMD32fAVX& a, const MLSIMD32fAVX& b)
  {
    return _mm256_div_ps(a.m_value, b.m_value);
  }

  // AVX 32 bit floating point division
  QM_ALWAYS_INLINE MLSIMD32cfAVX operator/(const MLSIMD32cfAVX& a, const MLSIMD32cfAVX& b)
  {
    __m256 mask = _mm256_set_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
    MLSIMD32cfAVX bconj = _mm256_xor_ps(b.m_value, mask);
    MLSIMD32cfAVX num = a * bconj;

    __m256 b2 = _mm256_mul_ps(b.m_value, b.m_value);
    __m256 b2s = _mm256_shuffle_ps(b2, b2, _MM_SHUFFLE(2,3,0,1));
    __m256 den = _mm256_add_ps(b2, b2s);

    return _mm256_div_ps(num.m_value, den);
  }

  // AVX 64 bit floating point division
  QM_ALWAYS_INLINE MLSIMD64fAVX operator/(const MLSIMD64fAVX& a, const MLSIMD64fAVX& b)
  {
    return _mm256_div_pd(a.m_value, b.m_value);
  }

  // AVX 64 bit floating point division
  QM_ALWAYS_INLINE MLSIMD64cfAVX operator/(const MLSIMD64cfAVX& a, const MLSIMD64cfAVX& b)
  {
    __m256d mask = _mm256_set_pd(-0.0f, 0.0f, -0.0f, 0.0f);
    MLSIMD64cfAVX bconj = _mm256_xor_pd(b.m_value, mask);
    MLSIMD64cfAVX num = a * bconj;

    __m256d b2 = _mm256_mul_pd(b.m_value, b.m_value);
    __m256d b2s = _mm256_shuffle_pd(b2, b2, 0b0101);
    __m256d den = _mm256_add_pd(b2, b2s);

    return _mm256_div_pd(num.m_value, den);
  }
#endif

#if defined(ML_MATH_AVX2)
  // AVX2 integer division (no vectorized version available)
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator/(const TMLSIMDintAVX2<SIMD>& a, const TMLSIMDintAVX2<SIMD>& b)
  {
    TMLDecaySIMDType_t<SIMD> res;
    for (size_t i = 0; i < TMLSIMDSize_v<SIMD>; i++)
      res[i] = (~a)[i] / (~b)[i];
    return res;
  }
#endif

}

#endif
