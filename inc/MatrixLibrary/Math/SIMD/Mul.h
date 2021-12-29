// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_SIMD_Mul_H_
#define ML_MATH_SIMD_Mul_H_

// Includes
#include "SIMD.h"

namespace ML
{
  
  namespace Internal
  {
    template<typename SIMD>
    class TMLSIMDDefaultMulHelper
    {
    public:
      QM_ALWAYS_INLINE TMLSIMDDefaultMulHelper(TMLDecaySIMDType_t<SIMD>& res, const TMLSIMD<SIMD>& a, const TMLSIMD<SIMD>& b)
        : m_res(res), m_a(a), m_b(b) {}
      QM_ALWAYS_INLINE void operator()(std::size_t i) { m_res[i] = (~m_a)[i] * (~m_b)[i]; }
      
    private:
      TMLDecaySIMDType_t<SIMD>& m_res;
      const TMLSIMD<SIMD>& m_a;
      const TMLSIMD<SIMD>& m_b;
    };
  }

  // default multiplication
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator*(const TMLSIMD<SIMD>& a, const TMLSIMD<SIMD>& b)
  {
    TMLDecaySIMDType_t<SIMD> res;
    Internal::TMLSIMDDefaultMulHelper<SIMD> helper(res, a, b);
    MLConstexprFor<std::size_t, 0, TMLSIMDSize_v<SIMD>, 1>(helper);
    return res;
  }

#if defined(ML_MATH_SSE)
  // SSE 32 bit floating point multiplication
  QM_ALWAYS_INLINE MLSIMD32fSSE operator*(const MLSIMD32fSSE& a, const MLSIMD32fSSE& b)
  {
    return _mm_mul_ps(a.m_value, b.m_value);
  }

  // SSE(3) 32 bit complex floating point multiplication
  QM_ALWAYS_INLINE MLSIMD32cfSSE operator*(const MLSIMD32cfSSE& a, const MLSIMD32cfSSE& b)
  {
    __m128 x, y, z;
    x = _mm_shuffle_ps(a.m_value, a.m_value, _MM_SHUFFLE(2,2,0,0));
    z = _mm_mul_ps(x, b.m_value);
    x = _mm_shuffle_ps(a.m_value, a.m_value, _MM_SHUFFLE(3,3,1,1));
    y = _mm_shuffle_ps(b.m_value, b.m_value, _MM_SHUFFLE(2,3,0,1));
    y = _mm_mul_ps(x, y);
#if defined(ML_MATH_SSE3)
    return _mm_addsub_ps(z, y);
#else
    __m128 mask = _mm_set_ps(-0.0f, 0.0f, -0.0f, 0.0f);
    y = _mm_xor_ps(y, mask);
    return _mm_add_ps(z, y);
#endif
  }
#endif

#if defined(ML_MATH_SSE2)
  // SSE 64 bit floating point multiplication
  QM_ALWAYS_INLINE MLSIMD64fSSE2 operator*(const MLSIMD64fSSE2& a, const MLSIMD64fSSE2& b)
  {
    return _mm_mul_pd(a.m_value, b.m_value);
  }

  // SSE2(3) 64 bit complex floating point multiplication
  QM_ALWAYS_INLINE MLSIMD64cfSSE2 operator*(const MLSIMD64cfSSE2& a, const MLSIMD64cfSSE2& b)
  {
    __m128d x, y, z;
    x = _mm_shuffle_pd(a.m_value, a.m_value, _MM_SHUFFLE2(0,0));
    z = _mm_mul_pd(x, b.m_value);
    x = _mm_shuffle_pd(a.m_value, a.m_value, _MM_SHUFFLE2(1,1));
    y = _mm_shuffle_pd(b.m_value, b.m_value, _MM_SHUFFLE2(0,1));
    y = _mm_mul_pd(x, y);
#if defined(ML_MATH_SSE3)
    return _mm_addsub_pd(z, y);
#else
    __m128 mask = _mm_set_pd(-0.0, 0.0);
    y = _mm_xor_pd(y, mask);
    return _mm_add_pd(z, y);
#endif
  }

  // SSE 16 bit integer multiplication
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator*(const TMLSIMD16uiSSE2<SIMD>& a, const TMLSIMD16uiSSE2<SIMD>& b)
  {
    return _mm_mullo_epi16((~a).m_value, (~b).m_value);
  }

  // SSE 16 bit complex integer multiplication
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator*(const TMLSIMD16cuiSSE2<SIMD>& a, const TMLSIMD16cuiSSE2<SIMD>& b)
  {
    __m128i x, y, z;
    x = _mm_shufflelo_epi16((~a).m_value, _MM_SHUFFLE(2,2,0,0));
    x = _mm_shufflehi_epi16(x, _MM_SHUFFLE(2,2,0,0));
    z = _mm_mullo_epi16(x, (~b).m_value);
    x = _mm_shufflelo_epi16((~a).m_value, _MM_SHUFFLE(3,3,1,1));
    x = _mm_shufflehi_epi16(x, _MM_SHUFFLE(3,3,1,1));
    y = _mm_shufflelo_epi16((~b).m_value, _MM_SHUFFLE(2,3,0,1));
    y = _mm_shufflehi_epi16(y, _MM_SHUFFLE(2,3,0,1));
    y = _mm_mullo_epi16(x, y);

    __m128i mask = _mm_set_epi16(0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000);
    z = _mm_add_epi16(z, _mm_and_si128(mask, y));
    return _mm_sub_epi16(z, _mm_andnot_si128(mask, y));
  }

#if defined(ML_MATH_SSE4_1)
  // SSE 32 bit integer multiplication
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator*(const TMLSIMD32uiSSE2<SIMD>& a, const TMLSIMD32uiSSE2<SIMD>& b)
  {
    return _mm_mullo_epi32(a.m_value, b.m_value);
  }

  // SSE 32 bit complex integer multiplication
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator*(const TMLSIMD32cuiSSE2<SIMD>& a, const TMLSIMD32cuiSSE2<SIMD>& b)
  {
    __m128i x, y, z;
    x = _mm_shuffle_epi32((~a).m_value, _MM_SHUFFLE(2,2,0,0));
    z = _mm_mullo_epi32(x, (~b).m_value);
    x = _mm_shuffle_epi32((~a).m_value, _MM_SHUFFLE(3,3,1,1));
    y = _mm_shuffle_epi32((~b).m_value, _MM_SHUFFLE(2,3,0,1));
    y = _mm_mullo_epi32(x, y);

    __m128i mask = _mm_set_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000);
    z = _mm_add_epi32(z, _mm_and_si128(mask, y));
    return _mm_sub_epi32(z, _mm_andnot_si128(mask, y));
  }
#endif
#endif

#if defined(ML_MATH_AVX)
  // AVX 32 bit floating point multiplication
  QM_ALWAYS_INLINE MLSIMD32fAVX operator*(const MLSIMD32fAVX& a, const MLSIMD32fAVX& b)
  {
    return _mm256_mul_ps(a.m_value, b.m_value);
  }

  // AVX 32 bit complex floating point multiplication
  QM_ALWAYS_INLINE MLSIMD32cfAVX operator*(const MLSIMD32cfAVX& a, const MLSIMD32cfAVX& b)
  {
    __m256 x, y, z;
    x = _mm256_shuffle_ps(a.m_value, a.m_value, _MM_SHUFFLE(2,2,0,0));
    z = _mm256_mul_ps(x, b.m_value);
    x = _mm256_shuffle_ps(a.m_value, a.m_value, _MM_SHUFFLE(3,3,1,1));
    y = _mm256_shuffle_ps(b.m_value, b.m_value, _MM_SHUFFLE(2,3,0,1));
    y = _mm256_mul_ps(x, y);
    return _mm256_addsub_ps(z, y);
  }

  // AVX 64 bit floating point multiplication
  QM_ALWAYS_INLINE MLSIMD64fAVX operator*(const MLSIMD64fAVX& a, const MLSIMD64fAVX& b)
  {
    return _mm256_mul_pd(a.m_value, b.m_value);
  }

  // AVX 64 bit complex floating point multiplication
  QM_ALWAYS_INLINE MLSIMD64cfAVX operator*(const MLSIMD64cfAVX& a, const MLSIMD64cfAVX& b)
  {
    __m256d x, y, z;
    x = _mm256_shuffle_pd(a.m_value, a.m_value, 0b0000);
    z = _mm256_mul_pd(x, b.m_value);
    x = _mm256_shuffle_pd(a.m_value, a.m_value, 0b1111);
    y = _mm256_shuffle_pd(b.m_value, b.m_value, 0b0101);
    y = _mm256_mul_pd(x, y);
    return _mm256_addsub_pd(z, y);
  }
#endif

#if defined(ML_MATH_AVX2)
  // AVX2 16 bit integer multiplication
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator*(const TMLSIMD16uiAVX2<SIMD>& a, const TMLSIMD16uiAVX2<SIMD>& b)
  {
    return _mm256_mullo_epi16((~a).m_value, (~b).m_value);
  }

  // AVX2 16 bit complex integer multiplication
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator*(const TMLSIMD16cuiAVX2<SIMD>& a, const TMLSIMD16cuiAVX2<SIMD>& b)
  {
    __m256i x, y, z;
    x = _mm256_shufflelo_epi16((~a).m_value, _MM_SHUFFLE(2,2,0,0));
    x = _mm256_shufflehi_epi16(x, _MM_SHUFFLE(2,2,0,0));
    z = _mm256_mullo_epi16(x, (~b).m_value);
    x = _mm256_shufflelo_epi16((~a).m_value, _MM_SHUFFLE(3,3,1,1));
    x = _mm256_shufflehi_epi16(x, _MM_SHUFFLE(3,3,1,1));
    y = _mm256_shufflelo_epi16((~b).m_value, _MM_SHUFFLE(2,3,0,1));
    y = _mm256_shufflehi_epi16(y, _MM_SHUFFLE(2,3,0,1));
    y = _mm256_mullo_epi16(x, y);

    __m256i mask = _mm256_set_epi16(
      0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000,
      0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000);
    z = _mm256_add_epi16(z, _mm256_and_si256(mask, y));
    return _mm256_sub_epi16(z, _mm256_andnot_si256(mask, y));
  }

  // AVX2 32 bit integer multiplication
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator*(const TMLSIMD32uiAVX2<SIMD>& a, const TMLSIMD32uiAVX2<SIMD>& b)
  {
    return _mm256_mullo_epi32(a.m_value, b.m_value);
  }

  // AVX2 32 bit complex integer multiplication
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    operator*(const TMLSIMD32cuiAVX2<SIMD>& a, const TMLSIMD32cuiAVX2<SIMD>& b)
  {
    __m256i x, y, z;
    x = _mm256_shuffle_epi32((~a).m_value, _MM_SHUFFLE(2,2,0,0));
    z = _mm256_mullo_epi32(x, (~b).m_value);
    x = _mm256_shuffle_epi32((~a).m_value, _MM_SHUFFLE(3,3,1,1));
    y = _mm256_shuffle_epi32((~b).m_value, _MM_SHUFFLE(2,3,0,1));
    y = _mm256_mullo_epi32(x, y);

    __m256i mask = _mm256_set_epi32(
      0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
      0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000);
    z = _mm256_add_epi32(z, _mm256_and_si256(mask, y));
    return _mm256_sub_epi32(z, _mm256_andnot_si256(mask, y));
  }
#endif

}

#endif
