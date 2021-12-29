// Copyright 2021, Philipp Neufeld

#ifndef ML_Benchmark_H_
#define ML_Benchmark_H_

// Includes
#include <cstdint>
#include <type_traits>
#include <functional>
#include <tuple>
#include <chrono>

#include "../Core/Interface.h"
#include "DoNotOptimizeAway.h"
#include "FunctionTypeTraits.h"
#include "../QTL/EnableIf.h"

namespace ML
{

  interface IMLBenchmarkExperiment : TMLInterface<IMLUnknown,
    MLMakeUUID("EF266832-F28A-4A8E-9B47-C7C00C7F98BE")>
  {
    virtual void RunMeasurement(int iters) = 0;
    virtual std::uintmax_t GetMeasuredIterationCount() = 0;
    virtual std::uintmax_t GetMeasuredTime() = 0;
  };
  MLDefineIID(IMLBenchmarkExperiment);

  interface IMLBenchmark : TMLInterface<IMLUnknown,
    MLMakeUUID("E1636C47-97EA-4395-92AF-BBB09BE7F677")>
  {
    virtual std::size_t AddExperiment(
      IMLBenchmarkExperiment* pExperiment,
      const char* szExperimentName) = 0;
    virtual void RunBenchmark() = 0;

    virtual double GetExperimentResult(std::size_t experimentId) = 0;
    virtual void PrintResults() = 0;
  };
  MLDefineIID(IMLBenchmark);


  /* To benchmark a function you can use one of the following methods:
   *
   *		1. wrap in function: void func() { code to benchmark }
   *		   Usage: This method works well for macro benchmarks above
   *        approx. 1 microseconds
   *
   *		2. wrap in function and return number of internal iterations:
   *		   int func() { for(...) { code to benchmark } return iters; }
   *		   Usage: This method works well for macro benchmarks (see 1)
   *          and if the number of internal iterations is sufficiently high
   *          it can also be used for microbenchmarks
   *
   *		3. wrap in function that takes the number of iterations as a parameter
   *		   and returns the actual internal iteration count:
   *		   int func(int iters) { for(iters) { benchmark code } return iters; }
   *		   Usage: This method works for macro benchmarks as well
   *        as for micro benchmarks. If you are planning to do a micro
   *        benchmark this should be the preferred method.
   */

  namespace Internal
  {

    template<typename Lambda>
    struct TMLIsBenchmarkFunctionType1
    {
      constexpr static bool value =
        TMLGetFunctionParameterCount<Lambda>::value == 0 &&
        std::is_same<
        typename TMLGetFunctionReturnType<Lambda>::type, void
        >::value;
    };

    template<typename Lambda>
    struct TMLIsBenchmarkFunctionType2
    {
      constexpr static bool value =
        TMLGetFunctionParameterCount<Lambda>::value == 0 &&
        std::is_integral<
        typename TMLGetFunctionReturnType<Lambda>::type
        >::value;
    };


    template<bool has1Parameter, typename Lambda>
    struct TMLIsBenchmarkFunctionType3Helper
    {
      static constexpr bool value =
        std::is_integral<
        typename TMLTypelistFront<
        typename TMLGetFunctionParameterList<Lambda>::type
        >::type
        >::value &&
        sizeof(typename TMLTypelistFront<
          typename TMLGetFunctionParameterList<Lambda>::type
        >::type) >= sizeof(int);
    };
    template<typename Lambda>
    struct TMLIsBenchmarkFunctionType3Helper<false, Lambda>
    {
      static constexpr bool value = false;
    };

    template<typename Lambda>
    struct TMLIsBenchmarkFunctionType3
    {
      constexpr static bool value =
        TMLIsBenchmarkFunctionType3Helper<
        TMLGetFunctionParameterCount<Lambda>::value == 1, Lambda>::value &&
        std::is_integral<typename TMLGetFunctionReturnType<Lambda>::type>::value;
    };

    template<typename Lambda>
    struct TMLIsBenchmarkFunctionTypeAny
    {
      constexpr static bool value =
        TMLIsBenchmarkFunctionType1<Lambda>::value ||
        TMLIsBenchmarkFunctionType2<Lambda>::value ||
        TMLIsBenchmarkFunctionType3<Lambda>::value;
    };
  }

  class CMLBenchmarkExperiment : public TMLClass<IMLBenchmarkExperiment>
  {
    using TimeIterPair = std::tuple<std::uintmax_t, std::uintmax_t>;
    using BenchmarkFunc = std::function<TimeIterPair(int)>;

  public:
    CMLBenchmarkExperiment() { SetFunction([]() {}); m_result = { 0, 0 }; }
    ~CMLBenchmarkExperiment() { }

    virtual void RunMeasurement(int iters) override
    {
      m_result = m_func(iters);
    }

    virtual std::uintmax_t GetMeasuredIterationCount() override
    {
      return std::get<1>(m_result);
    }

    virtual std::uintmax_t GetMeasuredTime() override
    {
      return std::get<0>(m_result);
    }

    template <class Lambda>
    inline typename TMLEnableIf<
      Internal::TMLIsBenchmarkFunctionTypeAny<Lambda>::value, void
    >::type SetFunction(Lambda lambda)
    {
      m_func = MakeBenchmarkFunction(lambda);
    }

  private:

    // accepts functions like: void func();
    template <class Lambda>
    inline typename TMLEnableIf<
      Internal::TMLIsBenchmarkFunctionType1<Lambda>::value, BenchmarkFunc
    >::type MakeBenchmarkFunction(Lambda func)
    {
      return MakeBenchmarkFunction([=](int iters) -> std::uintmax_t
        {
          std::uintmax_t totalIters = static_cast<std::uintmax_t>(iters);
          while (iters-- > 0) { func(); }
          return totalIters;
        });
    }

    // accepts functions like: int func();
    template <class Lambda>
    inline typename TMLEnableIf<
      Internal::TMLIsBenchmarkFunctionType2<Lambda>::value, BenchmarkFunc
    >::type MakeBenchmarkFunction(Lambda func)
    {
      return MakeBenchmarkFunction([=](int iters) -> std::uintmax_t
        {
          std::uintmax_t niter = 0;
          while (iters-- > 0) { niter += func(); }
          return niter;
        });
    }

    // accepts functions like: int func(int iters);
    template <class Lambda>
    inline typename TMLEnableIf<
      Internal::TMLIsBenchmarkFunctionType3<Lambda>::value, BenchmarkFunc
    >::type MakeBenchmarkFunction(Lambda func)
    {
      return [=](int iters) -> TimeIterPair
      {
        using IterArgType = typename TMLTypelistFront<
          typename TMLGetFunctionParameterList<Lambda>::type>::type;

        // measure the time it takes the function to run
        auto startTimeStamp = std::chrono::high_resolution_clock::now();
        auto internalIters_ret = func(static_cast<IterArgType>(iters));
        auto endTimeStamp = std::chrono::high_resolution_clock::now();

        // Return a tuple containing (dt, internalIters_ret)
        return TimeIterPair(
          std::chrono::nanoseconds(endTimeStamp - startTimeStamp).count(),
          static_cast<std::uintmax_t>(internalIters_ret));
      };
    }

  private:
    BenchmarkFunc m_func;
    TimeIterPair m_result;
  };

  template<typename Lambda>
  inline typename TMLEnableIf<
    Internal::TMLIsBenchmarkFunctionTypeAny<Lambda>::value,
    TMLInterfacePtr<IMLBenchmarkExperiment>
  >::type MLCreateBenchmarkExperimentObject(Lambda func)
  {
    CMLBenchmarkExperiment* pObj = new CMLBenchmarkExperiment;
    pObj->SetFunction(func);
    TMLInterfacePtr<IMLBenchmarkExperiment> pExperiment;
    pExperiment.Attach(static_cast<IMLBenchmarkExperiment*>(pObj));
    return pExperiment;
  }

}

#endif // !ML_Benchmark_H_
