// Copyright 2021, Philipp Neufeld

#ifndef ML_MATH_SIMD_FMA_H_
#define ML_MATH_SIMD_FMA_H_

// Includes
#include "SIMD.h"

#include "Add.h"
#include "Mul.h"

namespace ML
{
  
  // default fmadd
  template<typename SIMD>
  QM_ALWAYS_INLINE SIMD
    MLSIMDFmadd(const TMLSIMD<SIMD>& m1, const TMLSIMD<SIMD>& m2, const TMLSIMD<SIMD>& a)
  {
    return (~m1) * (~m2) + (~a);
  }

#if defined(ML_MATH_FMA) && defined(ML_MATH_SSE)
  // SSE 32 bit floating point multiplication
  QM_ALWAYS_INLINE MLSIMD32fSSE 
    MLSIMDFmadd(const MLSIMD32fSSE& m1, const MLSIMD32fSSE& m2, const MLSIMD32fSSE& a)
  {
    return _mm_fmadd_ps((~m1).m_value, (~m2).m_value, (~a).m_value);
  }
#endif

#if defined(ML_MATH_FMA) && defined(ML_MATH_AVX)
  // AVX 32 bit floating point multiplication
  QM_ALWAYS_INLINE MLSIMD32fAVX 
    MLSIMDFmadd(const MLSIMD32fAVX& m1, const MLSIMD32fAVX& m2, const MLSIMD32fAVX& a)
  {
    return _mm256_fmadd_ps((~m1).m_value, (~m2).m_value, (~a).m_value);
  }
#endif

}

#endif
