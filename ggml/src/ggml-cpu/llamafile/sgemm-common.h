// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"

#if defined(__MMA__)

typedef vector unsigned char vec_t;
typedef __vector_quad acc_t;

namespace mma_common {

// Common utility function for debugging accumulators
static inline void dump_acc(acc_t * acc, vector unsigned char* vec_C) {
    __builtin_mma_disassemble_acc(vec_C, acc);
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            printf("%-12.4f ", *((float*)&vec_C[j] + i));
        }
        printf("\n");
    }
}

// Common vector permutation and store function
static inline void vector_permute_store(vec_t *c, int numVec, unsigned char *vecOffset) {
    vec_t t[8], s[8];
    vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
    vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
    vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

    if (numVec == 2) {
        t[0] = vec_perm(c[0], c[1], swiz1);
        t[1] = vec_perm(c[2], c[3], swiz1);
        s[0] = vec_perm(t[0], t[1], swiz3);
        s[1] = vec_perm(t[0], t[1], swiz4);
        vec_xst(s[0], 0, (vec_t*)vecOffset);
        vec_xst(s[1], 0, (vec_t*)(vecOffset + 16));
    } else if (numVec == 4) {
        t[0] = vec_perm(c[0], c[1], swiz1);
        t[1] = vec_perm(c[0], c[1], swiz2);
        t[2] = vec_perm(c[2], c[3], swiz1);
        t[3] = vec_perm(c[2], c[3], swiz2);
        s[0] = vec_perm(t[0], t[2], swiz3);
        s[1] = vec_perm(t[0], t[2], swiz4);
        s[2] = vec_perm(t[1], t[3], swiz3);
        s[3] = vec_perm(t[1], t[3], swiz4);
        for (int i = 0; i < 4; ++i)
            vec_xst(s[i], 0, (vec_t*)(vecOffset + i * 16));
    } else if (numVec == 8) {
        for (int i = 0; i < 4; i += 2) {
            t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
            t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
        }
        for (int i = 4; i < 8; i += 2) {
            t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
            t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
        }
        s[0] = vec_perm(t[0], t[2], swiz3);
        s[1] = vec_perm(t[0], t[2], swiz4);
        s[2] = vec_perm(t[1], t[3], swiz3);
        s[3] = vec_perm(t[1], t[3], swiz4);
        s[4] = vec_perm(t[4], t[6], swiz3);
        s[5] = vec_perm(t[4], t[6], swiz4);
        s[6] = vec_perm(t[5], t[7], swiz3);
        s[7] = vec_perm(t[5], t[7], swiz4);
        vec_xst(s[0], 0, (vec_t*)(vecOffset + 0 * 16));
        vec_xst(s[4], 0, (vec_t*)(vecOffset + 1 * 16));
        vec_xst(s[1], 0, (vec_t*)(vecOffset + 2 * 16));
        vec_xst(s[5], 0, (vec_t*)(vecOffset + 3 * 16));
        vec_xst(s[2], 0, (vec_t*)(vecOffset + 4 * 16));
        vec_xst(s[6], 0, (vec_t*)(vecOffset + 5 * 16));
        vec_xst(s[3], 0, (vec_t*)(vecOffset + 6 * 16));
        vec_xst(s[7], 0, (vec_t*)(vecOffset + 7 * 16));
    }
}

template<typename T>
static inline void packNormal(const T* a, int64_t lda, int rows, int cols, unsigned char* vec) {
    int64_t i, j;
    T *aoffset = NULL;
    unsigned char *vecOffset = NULL;
    T * aoffsets[8];
    vector unsigned char c_arr[8];
    aoffset = const_cast<T*>(a);
    vecOffset = vec;
    j = (rows >> 3);
    if (j > 0) {
        do {
            aoffsets[0] = aoffset;
            if (cols == 4) {
                for (int it = 1; it < 4; ++it)
                    aoffsets[it] = aoffsets[it-1] + lda;
                aoffset += 4 * lda;
                for (int i = 0; i < 4; ++i)
                    c_arr[i] = vec_xl(0, (vector unsigned char*)aoffsets[i]);
                vector_permute_store(c_arr, 4, vecOffset);
                for (int i = 0; i<4; i++)
                    aoffsets[i] = aoffsets[i]+lda;
                vecOffset +=64;
            }
            i = (cols >> 3);
            if (i > 0) {
                for (int it = 1; it < 8; ++it) {
                    aoffsets[it] = aoffsets[it-1] + lda;
                }
                aoffset += 8 * lda;
                do {
                    for (int it = 0; it < 8; ++it)
                        c_arr[it] = vec_xl(0, (vector unsigned char*)aoffsets[it]);
                    vector_permute_store(c_arr, 8, vecOffset);
                    for (int it = 0; it < 8; ++it)
                        aoffsets[it] = aoffsets[it] + 8*lda;
                    vecOffset += 128;
                    i--;
                } while(i > 0);
            }
            j--;
        } while(j > 0);
    }
    if (rows & 4) {
        aoffsets[0] = aoffset;
        for (int it = 1; it < 4; ++it)
            aoffsets[it] = aoffsets[it-1] + lda;
        aoffset += 4 * lda;
        if (cols == 4) {
            for (int it = 0; it < 4; ++it)
                c_arr[it] = vec_xl(0, (vector unsigned char*)aoffsets[it]);
            vector_permute_store(c_arr, 2, vecOffset);
            for (int it = 0; it< 4; it++)
                aoffsets[it] = aoffsets[it] + lda;
            vecOffset += 32;
        }
        i = (cols >> 3);
        if (i > 0) {
            do {
                for (int it = 0; it < 4; ++it)
                    c_arr[it] = vec_xl(0, (vector unsigned char*)aoffsets[it]);
                vector_permute_store(c_arr, 4, vecOffset);
                for (int it = 0; it< 4; it++)
                    aoffsets[it] = aoffsets[it] + 8*lda;
                vecOffset += 64;
                i--;
            } while(i > 0);
        }
    }
    if (rows & 3) {
        aoffsets[0] = aoffset;
        for (int it = 1; it < 4; ++it)
            aoffsets[it] = aoffsets[it-1] + lda;
        if (cols == 4) {
            switch(rows) {
                case 3: c_arr[2] = vec_xl(0, (vector unsigned char*)aoffsets[2]);
                case 2: c_arr[1] = vec_xl(0, (vector unsigned char*)aoffsets[1]);
                case 1: c_arr[0] = vec_xl(0, (vector unsigned char*)aoffsets[0]);
                    break;
            }
            vector_permute_store(c_arr, 2, vecOffset);
            for (int it = 0; it< 4; it++)
                 aoffsets[it] = aoffsets[it] + lda;
            vecOffset += 32;
        }
        i = (cols >> 3);
        if (i > 0) {
            do {
                switch(rows) {
                    case 3: c_arr[2] = vec_xl(0, (vector unsigned char*)aoffsets[2]);
                    case 2: c_arr[1] = vec_xl(0, (vector unsigned char*)aoffsets[1]);
                    case 1: c_arr[0] = vec_xl(0, (vector unsigned char*)aoffsets[0]);
                        break;
                }
                vector_permute_store(c_arr, 4, vecOffset);
                for (int it = 0; it <4; it++)
                     aoffsets[it] = aoffsets[it] + 8* lda;
                vecOffset += 64;
                i--;
            } while(i > 0);
        }
    }
}

// Common small GEMM function template
template<typename T, int RM, int RN>
static inline void gemm_small_impl(
    int64_t m0, int64_t m, int64_t n0, int64_t n,
    const T *A, int64_t lda,
    const T *B, int64_t ldb,
    float *C, int64_t ldc,
    int64_t k, int ith, int nth,
    void (*mma_outer_product)(acc_t*, vec_t, vec_t))
{
    int64_t ytiles = (m - m0) / RM;
    int64_t xtiles = (n - n0) / RN;
    int64_t tiles = xtiles * ytiles;
    int64_t duty = (tiles + nth - 1) / nth;
    int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > tiles)
        end = tiles;
    for (int64_t job = start; job < end; ++job) {
        int64_t ii = m0 + job / xtiles * RM;
        int64_t jj = n0 + job % xtiles * RN;
        vec_t vec_C[4];
        acc_t acc_0;
        __builtin_mma_xxsetaccz(&acc_0);
        vec_t vec_A[2], vec_B[2];
        for (int l=0; l<k; l+=4) {
            packNormal<T>(A+(ii*lda)+l, lda, RM, 4, (uint8_t*)vec_A);
            packNormal<T>(B+(jj*ldb)+l, ldb, RN, 4, (uint8_t*)vec_B);
            for (int x = 0; x<2; x++) {
                mma_outer_product(&acc_0, vec_A[x], vec_B[x]);
            }
        }
        __builtin_mma_disassemble_acc(vec_C, &acc_0);
        for (int I = 0; I < RM; I++) {
            for (int J = 0; J < RN; J++) {
                *((float*)(C+ii+((jj+J)*ldc)+I)) = *((float*)&vec_C[I]+J);
            }
        }
    }
}

// Common Mx8 GEMM function template
template<typename T, int RM>
static inline void gemm_Mx8_impl(
    int64_t m0, int64_t m, int64_t n0, int64_t n,
    const T *A, int64_t lda,
    const T *B, int64_t ldb,
    float *C, int64_t ldc,
    int64_t k, int ith, int nth,
    void (*mma_outer_product)(acc_t*, vec_t, vec_t))
{
    int RN = 8;
    int64_t ytiles = (m - m0) / RM;
    int64_t xtiles = (n - n0) / RN;
    int64_t tiles = xtiles * ytiles;
    int64_t duty = (tiles + nth - 1) / nth;
    int64_t start = duty * ith;
    int64_t end = start + duty;
    if (end > tiles)
        end = tiles;
    for (int64_t job = start; job < end; ++job) {
        int64_t ii = m0 + job / xtiles * RM;
        int64_t jj = n0 + job % xtiles * RN;
        vec_t vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        vec_t vec_A[4], vec_B[8];
        for (int l=0; l<k; l+=8) {
            packNormal<T>(A+(ii*lda)+l, lda, RM, 8, (uint8_t*)vec_A);
            packNormal<T>(B+(jj*ldb)+l, ldb, RN, 8, (uint8_t*)vec_B);
            for (int x = 0; x<4; x++) {
                mma_outer_product(&acc_0, vec_A[x], vec_B[2*x]);
                mma_outer_product(&acc_1, vec_A[x], vec_B[2*x+1]);
            }
        }
        __builtin_mma_disassemble_acc(vec_C, &acc_0);
        for (int I = 0; I < RM; I++) {
            for (int J = 0; J < 4; J++) {
                *((float*)(C+ii+((jj+J)*ldc)+I)) = *((float*)&vec_C[I]+J);
            }
        }
        __builtin_mma_disassemble_acc(vec_C, &acc_1);
        for (int I = 0; I < RM; I++) {
            for (int J = 0; J < 4; J++) {
                *((float*)(C+ii+((jj+4+J)*ldc)+I)) = *((float*)&vec_C[I]+J);
            }
        }
    }
}
}
#endif 
