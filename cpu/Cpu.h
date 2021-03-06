// Copyright 2021, Philipp Neufeld

#ifndef ML_Cpu_H_
#define ML_Cpu_H_

#include <string>

namespace ML
{

    struct SMLX86Registers
    {
      int eax;
      int ebx;
      int ecx;
      int edx;
    };

    class CMLCpu
    {
    public:

      CMLCpu();
      ~CMLCpu(); 

      std::string GetVendor() const { return m_szVendor; }
      std::string GetBrand() const { return m_szBrand; }

      // SIMD
      bool HasMMX() const { return CMLCpu::IsBitSet(m_f_1_edx, 23); }
      bool HasSSE() const { return CMLCpu::IsBitSet(m_f_1_edx, 25); }
      bool HasSSE2() const { return CMLCpu::IsBitSet(m_f_1_edx, 26); }
      bool HasSSE3() const { return CMLCpu::IsBitSet(m_f_1_ecx, 0); }
      bool HasSSSE3() const { return CMLCpu::IsBitSet(m_f_1_ecx, 9); }
      bool HasSSE4_1() const { return CMLCpu::IsBitSet(m_f_1_ecx, 19); }
      bool HasSSE4_2() const { return CMLCpu::IsBitSet(m_f_1_ecx, 20); }
        
      bool HasFMA() const { return CMLCpu::IsBitSet(m_f_1_ecx, 12); }

      // cpu must also support osxsave in order for avx to be usable
      // https://insufficientlycomplicated.wordpress.com/2011/11/07/detecting-intel-advanced-vector-extensions-avx-in-visual-studio/
      bool HasAVX() const { return CMLCpu::IsBitSet(m_f_1_ecx, 28) && CMLCpu::IsBitSet(m_f_1_ecx, 27); }
      bool HasAVX2() const { return CMLCpu::IsBitSet(m_f_7_ebx, 5) && CMLCpu::IsBitSet(m_f_1_ecx, 27); }

    private:
      static void Cpuid(SMLX86Registers* pRegisters);
      static bool IsBitSet(int flag, int bitNum) { return !!(flag & (1 << bitNum)); }

    private:
      char m_szVendor[0x10];	// 12 + zero terminator + padding
      char m_szBrand[0x40];	// 48 + zero terminator + padding

      int m_f_1_ecx;
      int m_f_1_edx;
      int m_f_7_ebx;
      int m_f_7_ecx;
      int m_f_81_ecx;
      int m_f_81_edx;
    };

}

#endif // !ML_LIB_Cpu_H_
