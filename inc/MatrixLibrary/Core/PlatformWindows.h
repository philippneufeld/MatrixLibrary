// Copyright 2021, Philipp Neufeld

// prevent direct inclusion of this file
#if !defined (ML_PLATFORM_INCLUDES_ALLOW)
#	error PlatformWindows.h cannot be included directly. Include Platform.h.
#elif !defined (ML_PLATFORM_WINDOWS)
#	error PlatformWindows.h can only be included on windows builds.
#endif // !ML_PLATFORM_WINDOWS

#ifndef ML_PlatformWindows_H_
#define ML_PlatformWindows_H_

// cyclic dependence -> first include Plattform.h 
// -> includes PlatforWindows.h -> Plattform.h (hits include guard)
#include "Platform.h"

// check if required definitions already exist and define otherwise
#ifndef ML_DLL_PREFIX
#	define ML_DLL_PREFIX ""
#endif // !ML_DLL_PREFIX

#ifndef ML_DLL_SUFFIX
#	define ML_DLL_SUFFIX ".dll"
#endif // !ML_DLL_SUFFIX

// prevent Windows.h from defining macros for min and max (clashes with std c++)
#ifndef NOMINMAX	
# define NOMINMAX
#endif // !NOMINMAX

// prevent Windows.h from including unnecessary sub-headers 
// (speeds up compilation)
#ifndef WIN32_LEAN_AND_MEAN	
# define WIN32_LEAN_AND_MEAN
#endif // !WIN32_LEAN_AND_MEAN

#endif // !ML_PlatformWindows_H_
