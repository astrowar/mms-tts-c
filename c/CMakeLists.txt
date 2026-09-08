cmake_minimum_required(VERSION 3.10)
project(mms-tts C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# Options
# ============================================================
option(ENABLE_DUMP "Build with validation dump support" OFF)
option(ENABLE_OMP  "Build with OpenMP parallelism" ON)

# ============================================================
# Architecture detection
# ============================================================
include(CheckCCompilerFlag)

set(OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_base.c")
set(INT8_OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_int8.c")
set(Q16_OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_q16.c")
set(ARCH_FLAGS "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386")
    check_c_compiler_flag("-mavx2 -mfma" HAS_AVX2)
    if(HAS_AVX2)
        set(OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_avx.c")
        set(INT8_OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_int8_avx2.c")
        set(Q16_OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_q16_avx2.c")
        set(ARCH_FLAGS "-mavx2;-mfma")
        message(STATUS "Architecture: x86_64 + AVX2/FMA")
    else()
        message(STATUS "Architecture: x86_64 (no AVX2, using base)")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
    set(OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_neon.c")
    set(INT8_OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_int8_neon.c")
    message(STATUS "Architecture: ARM64 + NEON")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm")
    check_c_compiler_flag("-mfpu=neon" HAS_NEON)
    if(HAS_NEON)
        set(OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_neon.c")
        set(INT8_OPS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_int8_neon.c")
        set(ARCH_FLAGS "-mfpu=neon")
        message(STATUS "Architecture: ARMv7 + NEON")
    else()
        message(STATUS "Architecture: ARM (no NEON, using base)")
    endif()
else()
    message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR} (using base)")
endif()

# ============================================================
# Sources
# ============================================================
set(SRCS
    src/main.c
    ${OPS_SRC}
    ${INT8_OPS_SRC}
    ${Q16_OPS_SRC}
    src/tokenizer.c
    src/encoder.c
    src/duration.c
    src/flow.c
    src/hifigan_q.c
    src/hifigan_q16.c
    src/vtsm.c
    src/model.c
    src/wav.c
    src/npy_reader.c
)

if(ENABLE_DUMP)
    list(APPEND SRCS validate/dump.c)
endif()

# ============================================================
# Target
# ============================================================
add_executable(mms-tts ${SRCS})

target_compile_options(mms-tts PRIVATE
    -O2 -Wall -Wextra -Wno-unused-parameter
)

if(ENABLE_DUMP)
    target_compile_definitions(mms-tts PRIVATE ENABLE_DUMP)
endif()

target_link_libraries(mms-tts PRIVATE m)

# Architecture-specific flags apply to SIMD ops files
if(ARCH_FLAGS)
    set_source_files_properties(${OPS_SRC} PROPERTIES COMPILE_OPTIONS "${ARCH_FLAGS}")
    set_source_files_properties(${INT8_OPS_SRC} PROPERTIES COMPILE_OPTIONS "${ARCH_FLAGS}")
    set_source_files_properties(${Q16_OPS_SRC} PROPERTIES COMPILE_OPTIONS "${ARCH_FLAGS}")
endif()

# model_weights.h dependency
set_property(SOURCE src/vtsm.c APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/model_weights.h")

# ============================================================
# OpenMP
# ============================================================
if(ENABLE_OMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(mms-tts PRIVATE OpenMP::OpenMP_C)
    message(STATUS "OpenMP: enabled")
endif()

# ============================================================
# Install / summary
# ============================================================
message(STATUS "Build type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "Ops source:  ${OPS_SRC}")
message(STATUS "Int8 ops:    ${INT8_OPS_SRC}")
message(STATUS "Q16 ops:     ${Q16_OPS_SRC}")
message(STATUS "Dump:        ${ENABLE_DUMP}")
message(STATUS "OMP:         ${ENABLE_OMP}")
