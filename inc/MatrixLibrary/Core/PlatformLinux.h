// Copyright 2021, Philipp Neufeld

// prevent direct or incorrect inclusion of this file
#if !defined (ML_PLATFORM_INCLUDES_ALLOW)
#	error PlatformLinux.h cannot be included directly. Include Platform.h
#elif !defined (ML_PLATFORM_LINUX)
#	error PlatformLinux.h can only be included on linux builds.
#endif // !ML_PLATFORM_WINDOWS

// cyclic dependence -> first include Plattform.h 
// -> includes PlatformLinux.h -> Plattform.h (hits include guard)
#include "Platform.h"

#ifndef ML_PlatformLinux_H_
#define ML_PlatformLinux_H_

// check if required definitions already exist and define otherwise
#ifndef ML_DLL_PREFIX
#	define ML_DLL_PREFIX "lib"
#endif // !ML_DLL_PREFIX

#ifndef ML_DLL_SUFFIX
#	define ML_DLL_SUFFIX ".so"
#endif // !ML_DLL_SUFFIX

#endif // !ML_PlatformLinux
