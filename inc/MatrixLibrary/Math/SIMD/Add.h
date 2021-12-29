// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_SIMD_Add_H_
#define ML_MATH_SIMD_Add_H_

// Includes
#include "SIMD.h"

namespace ML
{

  namespace Internal
  {
    template<typename SIMD>
    class TMLSIMDDefaultAddHelper
    {
    public:
      QM_ALWAYS_INLINE TMLSIMDDefaultAddHelper(TMLDecaySIMDType_t<SIMD>& res, const TMLSIMD<SIMD>& a, const TMLSIMD<SIMD>& b)
        : m_res(res), m_a(a), m_b(b) {}
      QM_ALWAYS_INLINE void operator()(std::size_t i) { m_res[i] = (~m_a)[i] + (~m_b)[i]; }
      
    private:
      TMLDecaySIMDType_t<SIMD>& m_res;
      const TMLSIMD<SIMD>& m_a;
      const TMLSIMD<SIMD>& m_b;
    };
  }

  // default addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD<SIMD>& a, const TMLSIMD<SIMD>& b)
  {
    TMLDecaySIMDType_t<SIMD> res;
    Internal::TMLSIMDDefaultAddHelper<SIMD> helper(res, a, b);
    MLConstexprFor<std::size_t, 0, TMLSIMDSize_v<SIMD>, 1>(helper);
    return res;
  }

#if defined(ML_MATH_SSE)
  // SSE 32 bit floating point addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD32floatSSE<SIMD>& a, const TMLSIMD32floatSSE<SIMD>& b)
  {
    return _mm_add_ps((~a).m_value, (~b).m_value);
  }
#endif

#if defined(ML_MATH_SSE2)
  // SSE2 64 bit floating point addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD64floatSSE2<SIMD>& a, const TMLSIMD64floatSSE2<SIMD>& b)
  {
    return _mm_add_pd((~a).m_value, (~b).m_value);
  }

  // SSE2 8 bit integer addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD8intSSE2<SIMD>& a, const TMLSIMD8intSSE2<SIMD>& b)
  {
    return _mm_add_epi8((~a).m_value, (~b).m_value);
  }

  // SSE2 16 bit integer addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD16intSSE2<SIMD>& a, const TMLSIMD16intSSE2<SIMD>& b)
  {
    return _mm_add_epi16((~a).m_value, (~b).m_value);
  }

  // SSE2 32 bit integer addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD32intSSE2<SIMD>& a, const TMLSIMD32intSSE2<SIMD>& b)
  {
    return _mm_add_epi32((~a).m_value, (~b).m_value);
  }

  // SSE2 64 bit integer addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD64intSSE2<SIMD>& a, const TMLSIMD64intSSE2<SIMD>& b)
  {
    return _mm_add_epi64((~a).m_value, (~b).m_value);
  }
#endif

#if defined(ML_MATH_AVX)
  // AVX2 32 bit floating point addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD32floatAVX<SIMD>& a, const TMLSIMD32floatAVX<SIMD>& b)
  {
    return _mm256_add_ps((~a).m_value, (~b).m_value);
  }

  // AVX2 64 bit floating point addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD64floatAVX<SIMD>& a, const TMLSIMD64floatAVX<SIMD>& b)
  {
    return _mm256_add_pd((~a).m_value, (~b).m_value);
  }
#endif

#if defined(ML_MATH_AVX2)
  // AVX 8 bit integer addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD8intAVX2<SIMD>& a, const TMLSIMD8intAVX2<SIMD>& b)
  {
    return _mm256_add_epi8((~a).m_value, (~b).m_value);
  }
  
  // AVX2 16 bit integer addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD16intAVX2<SIMD>& a, const TMLSIMD16intAVX2<SIMD>& b)
  {
    return _mm256_add_epi16((~a).m_value, (~b).m_value);
  }

  // AVX2 32 bit integer addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD32intAVX2<SIMD>& a, const TMLSIMD32intAVX2<SIMD>& b)
  {
    return _mm256_add_epi32((~a).m_value, (~b).m_value);
  }

  // AVX2 64 bit integer addition
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD 
    operator+(const TMLSIMD64intAVX2<SIMD>& a, const TMLSIMD64intAVX2<SIMD>& b)
  {
    return _mm256_add_epi64((~a).m_value, (~b).m_value);
  }
#endif

}

#endif
