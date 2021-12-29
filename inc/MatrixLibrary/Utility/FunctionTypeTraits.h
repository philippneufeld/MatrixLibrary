// Copyright 2021, Philipp Neufeld

#ifndef ML_FunctionTypeTraits_H_
#define ML_FunctionTypeTraits_H_

// Includes
#include <type_traits>

#include "../QTL/EnableIf.h"
#include "../QTL/Boolean.h"
#include "../QTL/TypeList.h"

namespace ML
{

  template<typename T, typename = void>
  struct TMLIsCallable
    : std::is_function<typename std::remove_pointer<T>::type> { };

  template<typename T>
  struct TMLIsCallable<T, typename TMLEnableIf<
    std::is_same<
    decltype(void(&std::remove_pointer<T>::type::operator())), void
    >::value
  >::type> : std::true_type { };


  // Return type getter
  namespace Internal
  {

    template <typename Func>
    struct TMLGetFunctionReturnTypeImp
      : TMLGetFunctionReturnTypeImp<decltype(&Func::operator())> {};

    // standard function
    template<class R, class... Args>
    struct TMLGetFunctionReturnTypeImp<R(Args...)>
    {
      using type = R;
    };

    // function pointer
    template<class R, class... Args>
    struct TMLGetFunctionReturnTypeImp<R(*)(Args...)>
    {
      using type = R;
    };

    // function reference
    template<class R, class... Args>
    struct TMLGetFunctionReturnTypeImp<R(&)(Args...)>
    {
      using type = R;
    };

    // member function pointer
    template<class R, typename C, class... Args>
    struct TMLGetFunctionReturnTypeImp<R(C::*)(Args...)>
    {
      using type = R;
    };

    // const member function pointer
    template<class R, typename C, class... Args>
    struct TMLGetFunctionReturnTypeImp<R(C::*)(Args...) const>
    {
      using type = R;
    };

    template<typename Func, bool isFunction>
    struct TMLGetFunctionReturnTypeHelper
      : TMLGetFunctionReturnTypeImp<Func> { };
    template<typename Func>
    struct TMLGetFunctionReturnTypeHelper<Func, false>
    {
      using type = MLNoneType;
    };

    template<typename Func>
    struct TMLReduceFunctionType
    {
      using type = typename std::remove_pointer<
        typename std::remove_all_extents<Func>::type>::type;
    };
  }

  template<typename Func>
  struct TMLGetFunctionReturnType : Internal::TMLGetFunctionReturnTypeHelper<
    typename Internal::TMLReduceFunctionType<Func>::type,
    TMLIsCallable<typename Internal::TMLReduceFunctionType<Func>::type>::value
  > { };


  // Parameter list getter
  namespace Internal
  {
    template <typename Func>
    struct TMLGetFunctionParameterListImp
      : public TMLGetFunctionParameterListImp<decltype(&Func::operator())> { };

    // standard function
    template<class R, class... Args>
    struct TMLGetFunctionParameterListImp<R(Args...)>
    {
      using type = TMLTypelist<Args...>;
    };

    // function pointer
    template<class R, class... Args>
    struct TMLGetFunctionParameterListImp<R(*)(Args...)>
    {
      using type = TMLTypelist<Args...>;
    };

    // function reference
    template<class R, class... Args>
    struct TMLGetFunctionParameterListImp<R(&)(Args...)>
    {
      using type = TMLTypelist<Args...>;
    };

    // member function pointer
    template<class R, typename C, class... Args>
    struct TMLGetFunctionParameterListImp<R(C::*)(Args...)>
    {
      using type = TMLTypelist<Args...>;
    };

    // const member function pointer 
    template<class R, typename C, class... Args>
    struct TMLGetFunctionParameterListImp<R(C::*)(Args...) const>
    {
      using type = TMLTypelist<Args...>;
    };

    template<typename Func, bool isFunction>
    struct TMLGetFunctionParameterListHelper
      : TMLGetFunctionParameterListImp<Func> { };
    template<typename Func>
    struct TMLGetFunctionParameterListHelper<Func, false>
    {
      using type = TMLTypelist<>;
    };
  }

  template<typename Func>
  struct TMLGetFunctionParameterList
    : public Internal::TMLGetFunctionParameterListHelper<
    typename Internal::TMLReduceFunctionType<Func>::type,
    TMLIsCallable<typename Internal::TMLReduceFunctionType<Func>::type>::value
    > { };


  template<typename Func>
  struct TMLGetFunctionParameterCount
    : TMLTypelistLength<typename TMLGetFunctionParameterList<Func>::type> { };


  // Function signature comparator
  template <typename Func1, typename Func2>
  struct TMLIsSameFunctionSignature :
    TMLBooleanConstant<
    std::is_same<
    typename TMLGetFunctionReturnType<Func1>::type,
    typename TMLGetFunctionReturnType<Func2>::type
    >::value&&
    std::is_same<
    typename TMLGetFunctionParameterList<Func1>::type,
    typename TMLGetFunctionParameterList<Func2>::type
    >::value> { };

  namespace Internal
  {
    template <typename ParamListPrev, typename ParamList>
    struct TMLParameterList_RemoveConstRef_Imp;
    template <typename... PrevArgs, typename Arg, typename... Args>
    struct TMLParameterList_RemoveConstRef_Imp<
      TMLTypelist<PrevArgs...>, TMLTypelist<Arg, Args...>>
    {
      using type = typename TMLParameterList_RemoveConstRef_Imp<
        TMLTypelist<PrevArgs..., Arg>, TMLTypelist<Args...>>::type;
    };
    template <typename... PrevArgs, typename Arg, typename... Args>
    struct TMLParameterList_RemoveConstRef_Imp<
      TMLTypelist<PrevArgs...>, TMLTypelist<const Arg&, Args...>>
    {
      using type = typename TMLParameterList_RemoveConstRef_Imp<
        TMLTypelist<PrevArgs..., Arg>, TMLTypelist<Args...>>::type;
    };
    template <typename... PrevArgs, typename Arg, typename... Args>
    struct TMLParameterList_RemoveConstRef_Imp<
      TMLTypelist<PrevArgs...>, TMLTypelist<Arg&&, Args...>>
    {
      using type = typename TMLParameterList_RemoveConstRef_Imp<
        TMLTypelist<PrevArgs..., Arg>, TMLTypelist<Args...>>::type;
    };
    template <typename... PrevArgs, typename Arg, typename... Args>
    struct TMLParameterList_RemoveConstRef_Imp<
      TMLTypelist<PrevArgs...>, TMLTypelist<const Arg&&, Args...>>
    {
      using type = typename TMLParameterList_RemoveConstRef_Imp<
        TMLTypelist<PrevArgs..., Arg>, TMLTypelist<Args...>>::type;
    };
    template <typename... PrevArgs>
    struct TMLParameterList_RemoveConstRef_Imp<
      TMLTypelist<PrevArgs...>, TMLTypelist<>>
    {
      using type = TMLTypelist<PrevArgs...>;
    };

    template <typename ParamList>
    struct TMLParameterList_RemoveConstRef
      : public TMLParameterList_RemoveConstRef_Imp<TMLTypelist<>, ParamList>
    { };

    template<typename Func>
    struct TMLGetFunctionReducedParamList
    {
      using type = typename Internal::TMLParameterList_RemoveConstRef<
        typename TMLGetFunctionParameterList<Func>::type
      >::type;
    };
  }

  // Function signature comparator (identifies pass by value with pass 
  // by const reference (or (const) r-value reference))
  template <typename Func1, typename Func2>
  struct TMLIsSameFunctionSignature_IgnoreConstRef :
    TMLBooleanConstant<
    std::is_same<
    typename TMLGetFunctionReturnType<Func1>::type,
    typename TMLGetFunctionReturnType<Func2>::type
    >::value&&
    std::is_same<
    typename Internal::TMLGetFunctionReducedParamList<Func1>::type,
    typename Internal::TMLGetFunctionReducedParamList<Func2>::type
    >::value> { };


}

#endif // !ML_FunctionTypeTraits_H_
