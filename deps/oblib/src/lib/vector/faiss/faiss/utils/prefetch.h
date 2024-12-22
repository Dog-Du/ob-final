/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// prefetches

#ifdef __AVX__

// AVX

#include <xmmintrin.h>

inline void prefetch_L1(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T0);
}
inline void prefetch_L2(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T1);
}
inline void prefetch_L3(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T2);
}

#define prefetch(p, byteSizeOfObject)           \
    do {                                        \
        unsigned char* ptr = (unsigned char*)p; \
        switch ((byteSizeOfObject - 1) >> 6) {  \
            default:                            \
            case 28:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 27:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 26:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 25:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 24:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 23:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 22:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 21:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 20:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 19:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 18:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 17:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 16:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 15:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 14:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 13:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 12:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 11:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 10:                            \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 9:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 8:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 7:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 6:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 5:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 4:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 3:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 2:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 1:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
            case 0:                             \
                _mm_prefetch(ptr, _MM_HINT_T0); \
                ptr += 64;                      \
                break;                          \
        }                                       \
    } while (0)

#elif defined(__aarch64__)

// ARM64

#ifdef _MSC_VER

// todo: arm on MSVC
inline void prefetch_L1(const void* address) {}
inline void prefetch_L2(const void* address) {}
inline void prefetch_L3(const void* address) {}

#else
// arm on non-MSVC

inline void prefetch_L1(const void* address) {
    __builtin_prefetch(address, 0, 3);
}
inline void prefetch_L2(const void* address) {
    __builtin_prefetch(address, 0, 2);
}
inline void prefetch_L3(const void* address) {
    __builtin_prefetch(address, 0, 1);
}
#endif

#else

// a generic platform

#ifdef _MSC_VER

inline void prefetch_L1(const void* address) {}
inline void prefetch_L2(const void* address) {}
inline void prefetch_L3(const void* address) {}

#else

inline void prefetch_L1(const void* address) {
    __builtin_prefetch(address, 0, 3);
}
inline void prefetch_L2(const void* address) {
    __builtin_prefetch(address, 0, 2);
}
inline void prefetch_L3(const void* address) {
    __builtin_prefetch(address, 0, 1);
}

#endif

#endif
