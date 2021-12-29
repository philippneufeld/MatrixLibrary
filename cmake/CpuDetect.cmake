
include(CheckFlags)
 
# Detecting CPU
set(CMAKE_TRY_COMPILE_TARGET_TYPE, EXECUTABLE)
set(CPU_DETECT_BINARY "${CMAKE_BINARY_DIR}/CMakeFiles/CpuDetect${CMAKE_EXECUTABLE_SUFFIX}")
try_compile(CPU_DETECT_COMPILED 
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_SOURCE_DIR}/cpu/Cpu.cpp"
    OUTPUT_VARIABLE TRY_COMPILE_CPU_DETECT_OUTPUT
    COMPILE_DEFINITIONS "-DML_CPU_DETECTION"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${ML_INCLUDE_DIR}"
    COPY_FILE "${CPU_DETECT_BINARY}"
)

set(SIMD_COMPILER_FLAGS "")
set(SIMD_MACRO_DEFINITIONS "")

if(CPU_DETECT_COMPILED)
    message("Successfully compiled cpu detection.")

    execute_process(COMMAND "${CPU_DETECT_BINARY}"
        RESULT_VARIABLE CPU_DETECTION_RESULT
        OUTPUT_VARIABLE CPU_DETECTION_OUTPUT
        ERROR_QUIET
    )
    
    if (CPU_DETECTION_RESULT EQUAL 0)
        # Cpu detection has run successfully

        ############# FMA ##############
        set(ML_FMA_TEST_SNIPPET
            "__m128 x=_mm_set1_ps(0.5);x=_mm_fmadd_ps(x, x, x);"
        )

        if (CPU_DETECTION_OUTPUT MATCHES ".*Features:.* fma .*")
            set(ML_FMA_TEST_SOURCE 
                "#include<immintrin.h>
                int main(){__m128 x=_mm_set1_ps(0.5);x=_mm_fmadd_ps(x, x, x);return 0;}"
            )
            ml_select_flag(COMPILER_FLAG_FMA
            "${ML_FMA_TEST_SOURCE}" "${SIMD_COMPILER_FLAGS}" 
                "" "-mfma" "/arch:FMA"
            )
            set(SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS} ${COMPILER_FLAG_FMA}")
            set(SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS} ML_FMA=1")
            message("CPU support for FMA has been verified")
        endif()


        ############# SSE+AVX ##############     
        set(ML_SSE_TEST_SNIPPET
            "{auto x=_mm_set1_ps(0.5);}"
        )
        set(ML_SSE2_TEST_SNIPPET
            "${ML_SSE_TEST_SNIPPET};{auto x=_mm_set1_epi16(1);}"
        )
        set(ML_SSE3_TEST_SNIPPET
            "${ML_SSE2_TEST_SNIPPET};{auto x=_mm_set1_ps(0.5);x=_mm_moveldup_ps(x);}"
        )
        set(ML_SSSE3_TEST_SNIPPET 
            "${ML_SSE3_TEST_SNIPPET};{auto x=_mm_set1_epi32(1);x=_mm_abs_epi32(x);}"
        )
        set(ML_SSE4_1_TEST_SNIPPET
            "${ML_SSSE3_TEST_SNIPPET};{auto x=_mm_set1_ps(0.5);x=_mm_dp_ps(x,x,0x77);}"
        )
        set(ML_SSE4_2_TEST_SNIPPET
            "${ML_SSE4_1_TEST_SNIPPET};{auto x = _mm_crc32_u8(1, 1);}"
        )
        set(ML_AVX_TEST_SNIPPET
            "${ML_SSE4_2_TEST_SNIPPET};{auto x=_mm256_set1_epi32(1);}"
        )
        set(ML_AVX2_TEST_SNIPPET
            "${ML_AVX_TEST_SNIPPET};{auto x=_mm256_set1_epi32(1);x=_mm256_abs_epi32(x);}"
        )
        
        set(ML_AVX_SSE_COMPILER_FLAG_FOUND FALSE)

        # AVX 2 
        if (CPU_DETECTION_OUTPUT MATCHES ".*Features:.* avx2 .*")
            if (NOT ML_AVX_SSE_COMPILER_FLAG_FOUND)
                set(ML_AVX2_TEST_SOURCE 
                    "#include<immintrin.h>
                    int main(){${ML_AVX2_TEST_SNIPPET};return 0;}"
                )
                ml_select_flag(COMPILER_FLAG_AVX2 
                "${ML_AVX2_TEST_SOURCE}" "${SIMD_COMPILER_FLAGS}" 
                    "/arch:AVX2" "-mavx2"
                )
                set(SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS} ${COMPILER_FLAG_AVX2}")
                set(ML_AVX_SSE_COMPILER_FLAG_FOUND TRUE)
            endif()
            set(SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS} ML_AVX2=1")
            message("CPU support for AVX2 has been verified")
        endif()

        # AVX
        if (CPU_DETECTION_OUTPUT MATCHES ".*Features:.* avx .*")
            if (NOT ML_AVX_SSE_COMPILER_FLAG_FOUND)
                set(ML_AVX_TEST_SOURCE 
                    "#include<immintrin.h>
                    int main(){${ML_AVX_TEST_SNIPPET};return 0;}"
                )
                ml_select_flag(COMPILER_FLAG_AVX
                "${ML_AVX_TEST_SOURCE}" "${SIMD_COMPILER_FLAGS}" 
                    "/arch:AVX" "-mavx" 
                )
                set(SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS} ${COMPILER_FLAG_AVX}")
                set(ML_AVX_SSE_COMPILER_FLAG_FOUND TRUE)
            endif()
            set(SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS} ML_AVX=1")
            message("CPU support for AVX has been verified")
        endif()

        # SSE4.2 
        if (CPU_DETECTION_OUTPUT MATCHES ".*Features:.* sse4_2 .*")
            if (NOT ML_AVX_SSE_COMPILER_FLAG_FOUND)
                set(ML_SSE4_2_TEST_SOURCE 
                    "#include<immintrin.h>
                    int main(){${ML_SSE4_2_TEST_SNIPPET};return 0;}"
                )
                ml_select_flag(COMPILER_FLAG_SSE4_2
                "${ML_SSE4_2_TEST_SOURCE}" "${SIMD_COMPILER_FLAGS}" 
                    "" "-msse4.2" "/arch:SSE4.2"
                )
                set(SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS} ${COMPILER_FLAG_SSE4_2}")
                set(ML_AVX_SSE_COMPILER_FLAG_FOUND TRUE)
            endif()
            set(SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS} ML_SSE4_2=1")
            message("CPU support for SSE4.2 has been verified")
        endif()

        # SSE4.1 
        if (CPU_DETECTION_OUTPUT MATCHES ".*Features:.* sse4_1 .*")
            if (NOT ML_AVX_SSE_COMPILER_FLAG_FOUND)
                set(ML_SSE4_1_TEST_SOURCE 
                    "#include<immintrin.h>
                    int main(){${ML_SSE4_1_TEST_SNIPPET};return 0;}"
                )
                ml_select_flag(COMPILER_FLAG_SSE4_1
                "${ML_SSE4_1_TEST_SOURCE}" "${SIMD_COMPILER_FLAGS}" 
                    "" "-msse4.1" "/arch:SSE4.1"
                )
                set(SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS} ${COMPILER_FLAG_SSE4_1}")
                set(ML_AVX_SSE_COMPILER_FLAG_FOUND TRUE)
            endif()
            set(SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS} ML_SSE4_1=1")
            message("CPU support for SSE4.1 has been verified")
        endif()

        # SSSE3 
        if (CPU_DETECTION_OUTPUT MATCHES ".*Features:.* ssse3 .*")
            if (NOT ML_AVX_SSE_COMPILER_FLAG_FOUND)
                set(ML_SSSE3_TEST_SOURCE 
                    "#include<immintrin.h>
                    int main(){${ML_SSSE3_TEST_SNIPPET};return 0;}"
                )
                ml_select_flag(COMPILER_FLAG_SSSE3
                "${ML_SSSE3_TEST_SOURCE}" "${SIMD_COMPILER_FLAGS}" 
                    "" "-mssse3" "/arch:SSSE3"
                )
                set(SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS} ${COMPILER_FLAG_SSSE3}")
                set(ML_AVX_SSE_COMPILER_FLAG_FOUND TRUE)
            endif()
            set(SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS} ML_SSSE3=1")
            message("CPU support for SSSE3 has been verified")
        endif()

        # SSE3 
        if (CPU_DETECTION_OUTPUT MATCHES ".*Features:.* sse3 .*")
            if (NOT ML_AVX_SSE_COMPILER_FLAG_FOUND)
                set(ML_SSE3_TEST_SOURCE 
                    "#include<immintrin.h>
                    int main(){${ML_SSE3_TEST_SNIPPET};return 0;}"
                )
                ml_select_flag(COMPILER_FLAG_SSE3
                "${ML_SSE3_TEST_SOURCE}" "${SIMD_COMPILER_FLAGS}" 
                    "" "-msse3" "/arch:SSE3"
                )
                set(SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS} ${COMPILER_FLAG_SSE3}")
                set(ML_AVX_SSE_COMPILER_FLAG_FOUND TRUE)
            endif()
            set(SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS} ML_SSE3=1")
            message("CPU support for SSE3 has been verified")
        endif()

        # SSE2 
        if (CPU_DETECTION_OUTPUT MATCHES ".*Features:.* sse2 .*")
            if (NOT ML_AVX_SSE_COMPILER_FLAG_FOUND)
                set(ML_SSE2_TEST_SOURCE 
                    "#include<immintrin.h>
                    int main(){${ML_SSE2_TEST_SNIPPET};return 0;}"
                )
                ml_select_flag(COMPILER_FLAG_SSE2
                "${ML_SSE2_TEST_SOURCE}" "${SIMD_COMPILER_FLAGS}" 
                    "" "-msse2" "/arch:SSE2"
                )
                set(SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS} ${COMPILER_FLAG_SSE2}")
                set(ML_AVX_SSE_COMPILER_FLAG_FOUND TRUE)
            endif()
            set(SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS} ML_SSE2=1")
            message("CPU support for SSE2 has been verified")
        endif()

        # SSE 
        if (CPU_DETECTION_OUTPUT MATCHES ".*Features:.* sse .*")
            if (NOT ML_AVX_SSE_COMPILER_FLAG_FOUND)
                set(ML_SSE_TEST_SOURCE 
                    "#include<immintrin.h>
                    int main(){${ML_SSE_TEST_SNIPPET};return 0;}"
                )
                ml_select_flag(COMPILER_FLAG_SSE
                "${ML_SSE_TEST_SOURCE}" "${SIMD_COMPILER_FLAGS}" 
                    "" "-msse" "/arch:SSE"
                )
                set(SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS} ${COMPILER_FLAG_SSE}")
                set(ML_AVX_SSE_COMPILER_FLAG_FOUND TRUE)
            endif()
            set(SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS} ML_SSE=1")
            message("CPU support for SSE has been verified")
        endif()

        unset(ML_AVX_SSE_COMPILER_FLAG_FOUND)
        
        message("Vectorization compiler flags: ${SIMD_COMPILER_FLAGS}")

    else()
        message(FATAL_ERROR "Error occurred while running the cpu detection: ${CPU_DETECTION_OUTPUT}")
    endif()
else()
    message(FATAL_ERROR "Unable to compile cpu detection: ${TRY_COMPILE_CPU_DETECT_OUTPUT}")
endif()

# remove leading and trailing whitespaces
string(STRIP "${SIMD_COMPILER_FLAGS}" SIMD_COMPILER_FLAGS)
string(STRIP "${SIMD_MACRO_DEFINITIONS}" SIMD_MACRO_DEFINITIONS)

# make semicolon separated list
string (REPLACE " " ";" SIMD_COMPILER_FLAGS "${SIMD_COMPILER_FLAGS}")
string (REPLACE " " ";" SIMD_MACRO_DEFINITIONS "${SIMD_MACRO_DEFINITIONS}")
