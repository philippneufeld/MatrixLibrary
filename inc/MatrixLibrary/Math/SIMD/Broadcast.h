// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_SIMD_Broadcast_H_
#define ML_MATH_SIMD_Broadcast_H_

// Includes
#include <cassert>
#include "SIMD.h"

namespace ML
{
  
  
  namespace Internal
  {
    template<int idx, typename SIMD>
    class TMLSIMDDefaultBroadcastHelper
    {
    public:
      QM_ALWAYS_INLINE TMLSIMDDefaultBroadcastHelper(TMLDecaySIMDType_t<SIMD>& res, const TMLSIMD<SIMD>& v)
        : m_res(res), m_v(v) {}
      QM_ALWAYS_INLINE void operator()(std::size_t i) { m_res[i] = (~m_v)[idx]; }
      
    private:
      TMLDecaySIMDType_t<SIMD>& m_res;
      const TMLSIMD<SIMD>& m_v;
    };
  }

  // default division
  template<int idx, typename SIMD>
  QM_ALWAYS_INLINE SIMD MLSIMDBroadcast(const TMLSIMD<SIMD>& v)
  {
    TMLDecaySIMDType_t<SIMD> res;
    Internal::TMLSIMDDefaultBroadcastHelper<idx, SIMD> helper(res, v);
    MLConstexprFor<std::size_t, 0, TMLSIMDSize_v<SIMD>, 1>(helper);
    return res;
  }

#if defined(ML_MATH_SSE)
  // SSE 32 bit floating point broadcast
  template<int idx>
  QM_ALWAYS_INLINE MLSIMD32fSSE MLSIMDBroadcast(const MLSIMD32fSSE& v)
  {
    assert(idx < 4);
    return _mm_shuffle_ps(v.m_value, v.m_value, idx * 0x55);
  }

  // SSE 32 bit complex floating point broadcast
  template<int idx>
  QM_ALWAYS_INLINE MLSIMD32cfSSE MLSIMDBroadcast(const MLSIMD32cfSSE& v)
  {
    assert(idx < 2);
#if defined(ML_MATH_SSE2)
    __m128d reg = _mm_castps_pd(v.m_value);
    reg = _mm_shuffle_pd(reg, reg, idx * 0x03);
    return _mm_castpd_ps(reg);
#else
    return SIMD::Set1((~v)[idx]);
#endif
  }

#endif

#if defined(ML_MATH_SSE2)
  // SSE 64 bit floating point broadcast
  template<int idx>
  QM_ALWAYS_INLINE MLSIMD64fSSE2 MLSIMDBroadcast(const MLSIMD64fSSE2& v)
  {
    assert(idx < 2);
    return _mm_shuffle_pd(v.m_value, v.m_value, idx * 0x03);
  }

  // SSE2 32 bit integer broadcast
  template<int idx, typename SIMD>
  QM_ALWAYS_INLINE SIMD MLSIMDBroadcast(const TMLSIMD32uiSSE2<SIMD>& v)
  {
    assert(idx < 4);
    return _mm_shuffle_epi32((~v).m_value, idx * 0x55);
  }
#endif

#if defined(ML_MATH_AVX)
  // AVX 32 bit floating point broadcast
  template<int idx>
  QM_ALWAYS_INLINE MLSIMD32fAVX MLSIMDBroadcast(const MLSIMD32fAVX& v)
  {
    assert(idx < 8);
    __m256 reg = _mm256_permute2f128_ps(v.m_value, v.m_value, (idx >> 2) * 0x11);
    return _mm256_permute_ps(reg, (idx & 0x03) * 0x55);
  }

  // AVX 32 bit complex floating point broadcast
  template<int idx>
  QM_ALWAYS_INLINE MLSIMD32cfAVX MLSIMDBroadcast(const MLSIMD32cfAVX& v)
  {
    assert(idx < 4);
    __m256d reg = _mm256_castps_pd(v.m_value);
    reg = _mm256_permute2f128_pd(reg, reg, (idx >> 1) * 0x11);
    reg = _mm256_permute_pd(reg, (idx & 0x01) * 0x0f);
    return _mm256_castpd_ps(reg);
  }

  // AVX 64 bit floating point broadcast
  template<int idx>
  QM_ALWAYS_INLINE MLSIMD64fAVX MLSIMDBroadcast(const MLSIMD64fAVX& v)
  {
    assert(idx < 4);
    __m256d reg = _mm256_permute2f128_pd(v.m_value, v.m_value, (idx >> 1) * 0x11);
    return _mm256_permute_pd(reg, (idx & 0x01) * 0x0f);
  }

  // AVX 64 bit complex floating point broadcast
  template<int idx>
  QM_ALWAYS_INLINE MLSIMD64cfAVX MLSIMDBroadcast(const MLSIMD64cfAVX& v)
  {
    assert(idx < 2);
    return _mm256_permute2f128_pd(v.m_value, v.m_value, idx * 0x11);
  }
#endif

}

#endif
