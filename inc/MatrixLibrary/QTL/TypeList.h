// Copyright 2021, Philipp Neufeld

#ifndef ML_QTL_TypeList_H_
#define ML_QTL_TypeList_H_

// Includes
#include <cstddef> // std::size_t

namespace ML
{
  struct MLNoneType { };

  template <typename... Ts>
  struct TMLTypelist { };

  // length
  template <typename Typelist>
  struct TMLTypelistLength;
  template <typename... Ts>
  struct TMLTypelistLength<TMLTypelist<Ts...>>
  {
    constexpr static unsigned int value = sizeof...(Ts);
  };

  // append and prepend
  template <typename T, typename Typelist>
  struct TMLTypelistPrepend;
  template <typename T, typename... Ts>
  struct TMLTypelistPrepend<T, TMLTypelist<Ts...>>
  {
    using type = TMLTypelist<T, Ts...>;
  };

  template <bool cond, typename T, typename Typelist>
  struct TMLTypelistPrepend_if;
  template <typename T, typename... Ts>
  struct TMLTypelistPrepend_if<true, T, TMLTypelist<Ts...>>
  {
    using type = TMLTypelist<T, Ts...>;
  };
  template <typename T, typename... Ts>
  struct TMLTypelistPrepend_if<false, T, TMLTypelist<Ts...>>
  {
    using type = TMLTypelist<Ts...>;
  };

  template <typename T, typename Typelist>
  struct TMLTypelistAppend;
  template <typename T, typename... Ts>
  struct TMLTypelistAppend<T, TMLTypelist<Ts...>>
  {
    using type = TMLTypelist<Ts..., T>;
  };

  template <bool cond, typename T, typename Typelist>
  struct TMLTypelistAppend_if;
  template <typename T, typename... Ts>
  struct TMLTypelistAppend_if<true, T, TMLTypelist<Ts...>>
  {
    using type = TMLTypelist<Ts..., T>;
  };
  template <typename T, typename... Ts>
  struct TMLTypelistAppend_if<false, T, TMLTypelist<Ts...>>
  {
    using type = TMLTypelist<Ts...>;
  };

  // contains
  template <typename Ty, typename Typelist>
  struct TMLTypelistContains;
  template <typename Ty, typename T, typename... Ts>
  struct TMLTypelistContains<Ty, TMLTypelist<T, Ts...>>
  {
    constexpr static bool value =
      TMLTypelistContains<Ty, TMLTypelist<Ts...>>::value;
  };
  template <typename Ty, typename... Ts>
  struct TMLTypelistContains<Ty, TMLTypelist<Ty, Ts...>>
  {
    constexpr static bool value = true;
  };
  template <typename Ty>
  struct TMLTypelistContains<Ty, TMLTypelist<>>
  {
    constexpr static bool value = false;
  };

  template <typename TestTypelist, typename Typelist>
  struct TMLTypelistContainsTypelist;
  template <typename T, typename... Ts, typename... Us>
  struct TMLTypelistContainsTypelist<TMLTypelist<T, Ts...>, TMLTypelist<Us...>>
  {
    constexpr static bool value =
      TMLTypelistContains<T, TMLTypelist<Us...>>::value &&
      TMLTypelistContainsTypelist<TMLTypelist<Ts...>, TMLTypelist<Us...>>::value;
  };
  template <typename... Us>
  struct TMLTypelistContainsTypelist<TMLTypelist<>, TMLTypelist<Us...>>
  {
    constexpr static bool value = true;
  };

  // insert
  template <unsigned int idx, typename Ty, typename Typelist>
  struct TMLTypelistInsert;
  template <unsigned int idx, typename Ty, typename T, typename... Ts>
  struct TMLTypelistInsert<idx, Ty, TMLTypelist<T, Ts...>>
  {
    using type = typename TMLTypelistPrepend<
      T, typename TMLTypelistInsert<idx - 1, Ty, TMLTypelist<Ts...>>::type>::type;
  };
  template <typename Ty, typename T, typename... Ts>
  struct TMLTypelistInsert<0, Ty, TMLTypelist<T, Ts...>>
  {
    using type = TMLTypelist<Ty, T, Ts...>;
  };
  template <typename Ty>
  struct TMLTypelistInsert<0, Ty, TMLTypelist<>>
  {
    using type = TMLTypelist<Ty>;
  };

  template <bool cond, unsigned int idx, typename Ty, typename Typelist>
  struct TMLTypelistInsert_if;
  template <unsigned int idx, typename Ty, typename... Ts>
  struct TMLTypelistInsert_if<true, idx, Ty, TMLTypelist<Ts...>>
  {
    using type = typename TMLTypelistInsert<idx, Ty, TMLTypelist<Ts...>>::type;
  };
  template <unsigned int idx, typename Ty, typename... Ts>
  struct TMLTypelistInsert_if<false, idx, Ty, TMLTypelist<Ts...>>
  {
    using type = TMLTypelist<Ts...>;
  };

  // erase
  template <unsigned int idx, unsigned int cnt, typename Typelist>
  struct TMLTypelistEraseRange;
  template <unsigned int idx, unsigned int cnt, typename T, typename... Ts>
  struct TMLTypelistEraseRange<idx, cnt, TMLTypelist<T, Ts...>>
  {
    using type = typename TMLTypelistPrepend<
      T, typename TMLTypelistEraseRange<idx - 1, cnt, TMLTypelist<Ts...>>::type>::type;
  };
  template <unsigned int cnt, typename T, typename... Ts>
  struct TMLTypelistEraseRange<0, cnt, TMLTypelist<T, Ts...>>
  {
    using type = typename TMLTypelistEraseRange<
      0, cnt - 1, TMLTypelist<Ts...>>::type;
  };
  template <unsigned int idx, typename T, typename... Ts>
  struct TMLTypelistEraseRange<idx, 0, TMLTypelist<T, Ts...>>
  {
    using type = TMLTypelist<T, Ts...>;
  };
  template <unsigned int idx>
  struct TMLTypelistEraseRange<idx, 0, TMLTypelist<>>
  {
    using type = TMLTypelist<>;
  };
  template <typename T, typename... Ts>
  struct TMLTypelistEraseRange<0, 0, TMLTypelist<T, Ts...>>
  {
    using type = TMLTypelist<T, Ts...>;
  };
  template <>
  struct TMLTypelistEraseRange<0, 0, TMLTypelist<>>
  {
    using type = TMLTypelist<>;
  };

  template <unsigned int idx, typename Typelist>
  struct TMLTypelistErase
  {
    using type = typename TMLTypelistEraseRange<idx, 1, Typelist>::type;
  };
  template <typename Typelist>
  struct TMLTypelistEraseFront
  {
    using type = typename TMLTypelistEraseRange<0, 1, Typelist>::type;
  };
  template <typename Typelist>
  struct TMLTypelistEraseBack
  {
    using type = typename TMLTypelistEraseRange<
      TMLTypelistLength<Typelist>::value - 1, 1, Typelist>::type;
  };

  template <bool cond, unsigned int idx, typename Typelist>
  struct TMLTypelistErase_if;
  template <unsigned int idx, typename... Ts>
  struct TMLTypelistErase_if<true, idx, TMLTypelist<Ts...>>
  {
    using type = typename TMLTypelistErase<idx, TMLTypelist<Ts...>>::type;
  };
  template <unsigned int idx, typename... Ts>
  struct TMLTypelistErase_if<false, idx, TMLTypelist<Ts...>>
  {
    using type = TMLTypelist<Ts...>;
  };

  // element access
  template <unsigned int idx, typename Typelist>
  struct TMLTypelistGet;
  template <unsigned int idx, typename T, typename... Ts>
  struct TMLTypelistGet<idx, TMLTypelist<T, Ts...>>
  {
    using type = typename TMLTypelistGet<idx - 1, TMLTypelist<Ts...>>::type;
  };
  template <typename T, typename... Ts>
  struct TMLTypelistGet<0, TMLTypelist<T, Ts...>>
  {
    using type = T;
  };
  template <unsigned int idx>
  struct TMLTypelistGet<idx, TMLTypelist<>>
  {
    using type = MLNoneType; // fallback type
  };

  template <typename Typelist>
  struct TMLTypelistFront
  {
    using type = typename TMLTypelistGet<0, Typelist>::type;
  };

  template <typename Typelist>
  struct TMLTypelistBack
  {
    using type = typename TMLTypelistGet<
      TMLTypelistLength<Typelist>::value - 1, Typelist>::type;
  };

  // reverse
  template <typename Typelist>
  struct TMLTypelistReverse;
  template <typename T, typename... Ts>
  struct TMLTypelistReverse<TMLTypelist<T, Ts...>>
  {
    using type = typename TMLTypelistAppend<T,
      typename TMLTypelistReverse<TMLTypelist<Ts...>>::type>::type;
  };
  template <>
  struct TMLTypelistReverse<TMLTypelist<>>
  {
    using type = TMLTypelist<>;
  };

  // concatenate typelists
  template <typename Typelist1, typename Typelist2>
  struct TMLTypelistConcatenate;
  template <typename... Ts, typename... Us>
  struct TMLTypelistConcatenate<TMLTypelist<Ts...>, TMLTypelist<Us...>>
  {
    using type = TMLTypelist<Ts..., Us...>;
  };

  // remove duplicates
  template <typename Typelist>
  struct TMLTypelistRemoveDuplicates;
  template <typename... Us>
  struct TMLTypelistRemoveDuplicates<TMLTypelist<Us...>>
  {
  private:
    template <typename WorkingTypelist, typename Typelist>
    struct TMLTypelistRemoveDuplicates_impl;
    template <typename... Rs, typename T, typename... Ts>
    struct TMLTypelistRemoveDuplicates_impl<
      TMLTypelist<Rs...>, TMLTypelist<T, Ts...>>
    {
      using type = typename TMLTypelistRemoveDuplicates_impl<
        typename TMLTypelistAppend_if<
        !TMLTypelistContains<T, TMLTypelist<Rs...>>::value,
        T, TMLTypelist<Rs...>>::type,
        TMLTypelist<Ts...>>::type;
    };
    template <typename... Rs>
    struct TMLTypelistRemoveDuplicates_impl<TMLTypelist<Rs...>, TMLTypelist<>>
    {
      using type = TMLTypelist<Rs...>;
    };

  public:
    using type = typename TMLTypelistRemoveDuplicates_impl<
      TMLTypelist<>, TMLTypelist<Us...>>::type;
  };

}

#endif // !ML_VariadicTemplateList_H_
