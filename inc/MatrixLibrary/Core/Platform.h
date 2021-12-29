// Copyright 2021, Philipp Neufeld

#ifndef ML_Platform_H_
#define ML_Platform_H_

// compiler
#if defined(_MSC_VER) && !defined(__clang__)
#	ifndef ML_COMPILER_MSVC
#	define ML_COMPILER_MSVC
#	endif // !ML_COMPILER_MSVC
#elif defined(__clang__)
#	ifndef ML_COMPILER_CLANG
#	define ML_COMPILER_CLANG
#	endif // !ML_COMPILER_CLANG
#elif defined(__GNUC__)
#	ifndef ML_COMPILER_GNUC
#	define ML_COMPILER_GNUC
#	endif // !ML_COMPILER_GNUC
#else
#	ifndef ML_COMPILER_UNKNOWN
#	define ML_COMPILER_UNKNOWN
#	endif // !ML_COMPILER_UNKNOWN
#endif

// platform macro
#if defined (_WIN32)
#	ifndef ML_PLATFORM_WINDOWS
#	define ML_PLATFORM_WINDOWS
#	endif // !ML_PLATFORM_WINDOWS
#elif defined (__linux__)
#	ifndef ML_PLATFORM_LINUX
#	define ML_PLATFORM_LINUX
#	endif // !ML_PLATFORM_LINUX
#elif defined (__APPLE__)
#	ifndef ML_PLATFORM_MACOS
#	define ML_PLATFORM_MACOS
#	endif // !ML_PLATFORM_MACOS
#else
#	ifndef ML_PLATFORM_UNKNOWN
#	define ML_PLATFORM_UNKNOWN
#	endif // !ML_PLATFORM_UNKNOWN
#endif

// architecture
#if defined(ML_COMPILER_MSVC)
# if defined(_WIN64)
#   define ML_ARCH_X64
# else
#   define ML_ARCH_X32
# endif
#elif defined(ML_COMPILER_GNUC) || defined(ML_COMPILER_CLANG)
# if defined(__x86_64__) || defined(__ppc64)
#   define ML_ARCH_X64
# else
#   define ML_ARCH_X32
# endif
#endif

// output compiler message about platform
#if defined(ML_OUTPUT_PLATFORM_INFO)

#if defined(ML_COMPILER_MSVC)
#	pragma message("Compiler detected: Microsoft Visual Studio")
#elif defined(ML_COMPILER_CLANG)
#	pragma message("Compiler detected: Clang Compiler")
#elif defined(ML_COMPILER_GNUC)
#	pragma message("Compiler detected: GNU Compiler")
#else
#	pragma message("Compiler detection failed!")
#endif

#if defined(ML_PLATFORM_WINDOWS)
#	pragma message("Platform detected: Windows")
#elif defined(ML_PLATFORM_LINUX)
#	pragma message("Platform detected: Linux")
#elif defined(ML_PLATFORM_MACOS)
#	pragma message("Platform detected: Mac OS")
#else
#	pragma message("Platform detection failed!")
#endif

#endif // ML_OUTPUT_PLATFORM_INFO

// c++ version
#if defined ML_COMPILER_MSVC
# define ML_CPLUSPLUS _MSVC_LANG
#else
#	define ML_CPLUSPLUS __cplusplus
#endif

#if ML_CPLUSPLUS >= 201103L
#	define ML_HAS_CXX11
#endif

#if ML_CPLUSPLUS >= 201402L
#	define ML_HAS_CXX14
#endif 

#if ML_CPLUSPLUS >= 201703L
#	define ML_HAS_CXX17
#endif 

#if ML_CPLUSPLUS >= 202002L
#	define ML_HAS_CXX20
#endif 

// debug macros
#if defined (DEBUG) || defined (_DEBUG)
#	ifndef ML_DEBUG
#	define ML_DEBUG
#	endif
#endif

// miscellaneous
#ifndef NOEXCEPT
#	define NOEXCEPT noexcept
#endif

// include platform dependent header files
#ifdef ML_PLATFORM_INCLUDES_ALLOW
#	error ML_PLATFORM_INCLUDES_ALLOW should not be defined by the user
#endif
#define ML_PLATFORM_INCLUDES_ALLOW

#if defined (ML_PLATFORM_WINDOWS)
#	include "PlatformWindows.h"
#elif defined (ML_PLATFORM_LINUX)
#	include "PlatformLinux.h"
#else
#	error Unsupported platform. Please add Platform header file for your platform.
#endif // ML_PLATFORM_WINDOWS
#undef ML_PLATFORM_INCLUDES_ALLOW

// Platform header is required to define the following 
// macros if they are not already defined:
//  - ML_DLL_PREFIX
//  - ML_DLL_SUFFIX

// check if all required macros are defined
#if !defined(ML_DLL_PREFIX) || !defined(ML_DLL_SUFFIX)
#	error Plattform header does not define all the required macros
#endif

// shared library export specifier
#ifndef ML_DLL_EXPORT
#	if defined(_MSC_VER)
#		define ML_DLL_EXPORT __declspec(dllexport)
#	elif defined(__GNUC__)
//#		define ML_DLL_EXPORT __attribute__((dllexport))
#		define ML_DLL_EXPORT
#	else
#		define ML_DLL_EXPORT
#	endif
#endif // !ML_DLL_EXPORT

#ifndef ML_API
#	if !defined(ML_STATIC_LIB)
#		define ML_API extern "C" ML_DLL_EXPORT
#	else
#		define ML_API
#	endif
#endif // ML_DLL_API

#ifndef ML_EXPORT_API

#endif

namespace ML
{
  // define namespace
}

#endif // !ML_Platform_H_
