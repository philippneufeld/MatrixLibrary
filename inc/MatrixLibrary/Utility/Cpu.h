// Copyright 2021, Philipp Neufeld

#ifndef ML_Cpu_H_
#define ML_Cpu_H_

#include <MatrixLibrary/Core/Interface.h>

namespace ML
{
  interface IMLCpu : TMLInterface<IMLUnknown,
    MLMakeUUID("935255BC-AD3C-4BFE-9A5B-E89E43CD3333")>
  {
    virtual const char* GetVendor() const = 0;
    virtual const char* GetBrand() const = 0;

    // SIMD
    virtual bool HasMMX() const = 0;
    virtual bool HasSSE() const = 0;
    virtual bool HasSSE2() const = 0;
    virtual bool HasSSE3() const = 0;
    virtual bool HasSSSE3() const = 0;
    virtual bool HasSSE4_1() const = 0;
    virtual bool HasSSE4_2() const = 0;
    virtual bool HasFMA() const = 0;
    virtual bool HasAVX() const = 0;
    virtual bool HasAVX2() const = 0;
  };
  MLDefineIID(IMLCpu);
}

#endif // !ML_Cpu_H_
