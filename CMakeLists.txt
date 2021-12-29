# Copyright 2021, Philipp Neufeld
# 
# CMakeList.txt
# CMake definitions file for the project

cmake_minimum_required (VERSION 3.18)
message("CMAKE_VERSION = ${CMAKE_VERSION}")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "You don't want to configure in the source directory!")
endif()

# import cmake modules from ./cmake/
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# define project
project ("MatrixLibrary" VERSION 1.0)

# use C++14
set (CMAKE_CXX_STANDARD 14)

# Include directories
set(ML_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/inc;${CMAKE_SOURCE_DIR}/../blaze")

# CpuDetect checks for SIMD support and exports SIMD_MACRO_DEFINITIONS and SIMD_COMPILER_FLAGS
include (CpuDetect)

# Function that adds the CMakelist of an app and adds post build commands to copy the shared library
function(ml_add_app APP_TARGET)
    add_subdirectory("apps/${APP_TARGET}")

    # include directories
    target_include_directories("${APP_TARGET}" PUBLIC "${ML_INCLUDE_DIR}")

    # compiler definitions
    target_compile_definitions("${APP_TARGET}" PUBLIC "${SIMD_MACRO_DEFINITIONS}")

    # compiler flags and macro definitions
    target_compile_options("${APP_TARGET}" PUBLIC "${SIMD_COMPILER_FLAGS}")
endfunction()

# Add test apps
ml_add_app("Test")
