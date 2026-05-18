
#pragma once 

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "sgemm.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "simd-mappings.h"
#include "sgemm-common.h"

#include <array>
#include <type_traits>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#define MMA_PLUS 0

#if MMA_PLUS

typedef vector unsigned char vec_t;
typedef __vector_quad acc_t;

//typedef ggml_bf16_t T;

using namespace mma_common;

#define SAVE_ACC(ACC, ii, jj) \
   __builtin_mma_disassemble_acc(vec_C, ACC); \
   for (int I = 0; I < 4; I++) { \
      for (int J = 0; J < 4; J++) { \
         *((float*)(C+ii+((jj+J)*ldc)+I)) = *((float*)&vec_C[I]+J); \
      } \
   } \


template<typename T>
struct mma_instr;

template<>
struct mma_instr<ggml_bf16_t> {
    static inline void outer_product(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvbf16ger2pp(acc, a, b);
    }
};

template<>
struct mma_instr<ggml_fp16_t> {
    static inline void outer_product(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvf16ger2pp(acc, a, b);
    }
};
template<typename T>
struct mma_plus_instr;

template<>
struct mma_plus_instr<ggml_bf16_t> {
    static inline void outer_product(__dmr1024 *acc, __vector_pair* a, vec_t b) {
        __builtin_mma_dmxvbf16gerx2pp(acc, *a, b);
    }
};

template<>
struct mma_plus_instr<ggml_fp16_t> {
    static inline void outer_product(__dmr1024 *acc, __vector_pair* a, vec_t b) {
        __builtin_mma_dmxvf16gerx2pp(acc, *a, b);
    }
};

template<typename T>
class tinyBLAS_HP16_P12 {
  public:
    tinyBLAS_HP16_P12(int64_t k,
                const T *A, int64_t lda,
                const T *B, int64_t ldb,
                float*C, int64_t ldc,
                int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  private:
inline void save_dmr1024(__dmr1024* acc, int ii, int jj) {
    vec_t vec_C[8];
    __builtin_mma_disassemble_dmr(vec_C, acc);
    for (int I = 0; I < 8; I++) {
        int target_row = ii + (7 - I);
        for (int J = 0; J < 4; J++) {
            float* dest = (float *)(C + target_row + ((jj + J) * ldc));
            *dest = *((float *)&vec_C[I] + J);
        }
    }
}

    void dump_vec_bf16(const char * name, vec_t vec) {
    ggml_bf16_t* ptr = (ggml_bf16_t*)&vec;

    printf("%s:\t", name);
    for (int i = 0; i < 8; i++) {
        float val = ggml_compute_bf16_to_fp32(ptr[i]);
        printf("%-12.4f", val);
    }
    printf("\n");
}

    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        int m_rem = std::min<int64_t>(m - m0, 16);
        int n_rem = std::min<int64_t>(n - n0, 16);
	if (m_rem >= 16 && n_rem >= 16) {
		mc = 16;
		nc = 16;
		gemm<16,16>(m0, m, n0, n);
	} else if (m_rem >= 16 && n_rem >= 8) {
		mc = 16;
		nc = 8;
		gemm<16,8>(m0, m, n0, n);
	} else if (m_rem >= 16 && n_rem >= 4) { 
		mc = 16;
		nc = 4;
		gemm<16, 4>(m0, m, n0, n);
	}else if (m_rem >= 8 && n_rem >= 16) {
		mc = 8;
		nc = 16;
		gemm<8, 16>(m0, m, n0, n);
	} else 
        if (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } else if (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4,8>(m0, m, n0, n);
        } else if (m_rem >=8 && n_rem >=4){
                mc = 8;
                nc = 4;
                gemm<8,4>(m0, m, n0, n);
        } else if ((m_rem < 4) && (n_rem >= 8)) {
            nc = 8;
            switch(m_rem) {
                case 1:
                    mc = 1;
                    gemm_Mx8<1>(m0, m, n0, n);
                    break;
                case 2:
                    mc = 2;
                    gemm_Mx8<2>(m0, m, n0, n);
                    break;
                case 3:
                    mc = 3;
                    gemm_Mx8<3>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        } else if (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small<4, 4>(m0, m, n0, n);
        } else if ((m_rem > 4) && (n_rem < 4)) {
            mc = 4;
            switch(n_rem) {
                case 1:
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    break;
                case 2:
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    break;
                case 3:
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    break;

                default:
                    return;
            }
        } else {
            switch((m_rem << 4) | n_rem) {
                case 0x43:
                    mc = 4;
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    break;
                case 0x42:
                    mc = 4;
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    break;
                case 0x41:
                    mc = 4;
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    break;
                case 0x34:
                    mc = 3;
                    nc = 4;
                    gemm_small<3, 4>(m0, m, n0, n);
                    break;
                case 0x33:
                    mc = 3;
                    nc = 3;
                    gemm_small<3, 3>(m0, m, n0, n);
                    break;
                case 0x32:
                    mc = 3;
                    nc = 2;
                    gemm_small<3, 2>(m0, m, n0, n);
                    break;
                case 0x31:
                    mc = 3;
                    nc = 1;
                    gemm_small<3, 1>(m0, m, n0, n);
                    break;
                case 0x24:
                    mc = 2;
                    nc = 4;
                    gemm_small<2,4>(m0, m, n0, n);
                    break;
                case 0x23:
                    mc = 2;
                    nc = 3;
                    gemm_small<2, 3>(m0, m, n0, n);
                    break;
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm_small<2, 2>(m0, m, n0, n);
                    break;
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm_small<2, 1>(m0, m, n0, n);
                    break;
                case 0x14:
                    mc = 1;
                    nc = 4;
                    gemm_small<1, 4>(m0, m, n0, n);
                    break;
                case 0x13:
                    mc = 1;
                    nc = 3;
                    gemm_small<1, 3>(m0, m, n0, n);
                    break;
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm_small<1, 2>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm_small<1, 1>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    void KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        for (int l = 0; l < k; l+=8) {
            packNormal<T>((A+(ii*lda)+l), lda, 4, 8, (uint8_t*)vec_A);
            packNormal<T>((B+(jj*ldb)+l), ldb, 8, 8, (uint8_t*)vec_B);
            for (int x = 0; x < 4; x++) {
                mma_instr<T>::outer_product(&acc_0, vec_A[x], vec_B[2*x]);
                mma_instr<T>::outer_product(&acc_1, vec_A[x], vec_B[2*x +1]);
            }
        }
        SAVE_ACC(&acc_0, ii, jj);
        SAVE_ACC(&acc_1, ii, jj+4);
    }

    void KERNEL_8x4(int64_t ii, int64_t jj) {
    printf("IN MMA+ 8x4 kernel\n");
    vec_t vec_A[8], vec_B[4];
    __dmr1024 acc_0;
    __vector_pair vec_A0;

    __builtin_mma_dmsetdmrz(&acc_0);

    for (int l = 0; l < k; l += 8) {
        packNormal<T>((A + (ii * lda) + l), lda, 8, 8, (uint8_t*)vec_A);
        packNormal<T>((B + (jj * ldb) + l), ldb, 8, 4, (uint8_t*)vec_B);

        printf("\n--- DEBUG START: ii=%ld, jj=%ld, l=%d ---\n", ii, jj, l);

        for (int x = 0; x < 4; x++) {
            char labelA[32], labelB[32];
            sprintf(labelA, "  vec_A[%d]", x);
            sprintf(labelB, "  vec_B[%d]", x);

            // Debug Print vectors before MMA operation
            dump_vec_bf16(labelA, vec_A[x]);
            dump_vec_bf16(labelB, vec_B[x]);
            //vec_A0 = __builtin_vsx_lxvp(x * 32, (__vector_pair*)vec_A);
            __builtin_vsx_build_pair(&vec_A0 ,vec_A[2*x], vec_A[2*x+1]);
            mma_plus_instr<T>::outer_product(&acc_0, &vec_A0, vec_B[x]);
            //__builtin_mma_dmxvbf16gerx2pp(&acc_0, vec_A0, vec_B[x]);
            vector float debug_C[8]; 
            __builtin_mma_disassemble_dmr(debug_C, &acc_0);
            for (int i = 0; i < 8; ++i)
{
  union
  {
    __vector float v;
    float f[4];
  } u;
  u.v = debug_C[i];
  printf("Row %d: [%f, %f, %f, %f]\n",
         i,
         u.f[0],
         u.f[1],
         u.f[2],
         u.f[3]);
} 
 /*           printf("--- DMR Raw Integer Accumulator (ii: %ld, jj: %ld, l: %d) ---\n", ii, jj, l);
            for (int i = 0; i < 8; i++) {
            printf("Row %d: [%f, %f, %f, %f]\n", 
            i, ((float*)&debug_C[i])[0], ((float*)&debug_C[i])[1], ((float*)&debug_C[i])[2], ((float*)&debug_C[i])[3]);}*/
        }
        printf("--- DEBUG END ---\n");
    }

    // Write result back to memory
    save_dmr1024(&acc_0, ii, jj);
}

        void KERNEL_8x8(int64_t ii, int64_t jj) {
    printf("IN MMA+ 8x8 kernel\n");
    vec_t vec_A[8];
    vec_t vec_B[8];
    __dmr1024 acc_0, acc_1;
    __vector_pair vec_A0;

    __builtin_mma_dmsetdmrz(&acc_0);
    __builtin_mma_dmsetdmrz(&acc_1);

    for (int l = 0; l < k; l += 8) {
        // Use interleaved packing for A, regular for B
        packNormal<T>((A + (ii * lda) + l), lda, 8, 8, (uint8_t*)vec_A);
        packNormal<T>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t*)vec_B);

        printf("\n--- DEBUG START: ii=%ld, jj=%ld, l=%d ---\n", ii, jj, l);
        for (int it = 0; it < 8; it++) {
            char labelA[32], labelB[32];
            sprintf(labelA, "  vec_A[%d]", it);
            sprintf(labelB, "  vec_B[%d]", it);

            // Debug Print vectors before MMA operation
            dump_vec_bf16(labelA, vec_A[it]);
            dump_vec_bf16(labelB, vec_B[it]);
         }
        for (int x = 0; x < 4; x++) {
            // Direct load - vec_A is already in interleaved format!
            vec_A0 = __builtin_vsx_lxvp(x * 32, (__vector_pair*)vec_A);

            mma_plus_instr<T>::outer_product(&acc_0, &vec_A0, vec_B[2*x]);
            mma_plus_instr<T>::outer_product(&acc_1, &vec_A0, vec_B[2*x + 1]);
           // __builtin_mma_dmxvbf16gerx2pp(&acc[0], vec_A0, vec_B[2*x]);
            //__builtin_mma_dmxvbf16gerx2pp(&acc[1], vec_A0, vec_B[2*x+1]);

            printf ("printing acc0 at x = %d\n", x);
            vector float debug_C[8];
            __builtin_mma_disassemble_dmr(debug_C, &acc_0);
            for (int i = 0; i < 8; ++i) {
                 union
                {
                 __vector float v;
                float f[4];
                } u;
                u.v = debug_C[i];
                printf("Row %d: [%f, %f, %f, %f]\n", i, u.f[0], u.f[1], u.f[2], u.f[3]);
            }
        }
        printf("--- DEBUG END ---\n");
    }

    // Write result back to memory
    save_dmr1024(&acc_0, ii, jj);
    save_dmr1024(&acc_1, ii, jj+4);
}
    void KERNEL_16x16(int64_t ii, int64_t jj) {
    printf("In MMA + kernel 16x16\n");
    vec_t vec_A[16], vec_B[16]; 
    __vector_pair vec_A0, vec_A1;
    __dmr1024 acc[8];
    
    for(int i=0; i<8; i++) __builtin_mma_dmsetdmrz(&acc[i]);

    for (int l = 0; l < k; l += 8) {
        
        packNormal<T>(A + (ii * lda) + l, lda, 16, 8, (uint8_t*)vec_A);

        /*for (int i = 0; i < 4; i++) {
            vec_A[i*2]     = temp_A[i];     // Row 0, 1, 2, 3
            vec_A[i*2 + 1] = temp_A[i + 4]; // Row 4, 5, 6, 7
            
            vec_A[i*2 + 8] = temp_A[i + 8];  // Row 8, 9, 10, 11
            vec_A[i*2 + 9] = temp_A[i + 12]; // Row 12, 13, 14, 15
        }*/

        packNormal(B + (jj * ldb) + l, ldb, 8, 16, (uint8_t*)vec_B);
        /*printf("dumping input vectors after packing\n");
        for (int it = 0; it < 16; it ++) {
             char labelA[32], labelB[32];
            sprintf(labelA, "  vec_A[%d]", it);
            sprintf(labelB, "  vec_B[%d]", it);

            dump_vec_bf16(labelA, vec_A[it]);
            dump_vec_bf16(labelB, vec_B[it]);

        }*/
        // --- STEP 2: MMA EXECUTION WITH LXVP ---
        for (int x = 0; x < 4; x++) {
            // Load Vector Pairs directly from the interleaved memory
            // Each lxvp loads TWO 16-byte vectors into one 32-byte pair
            vec_A0 = __builtin_vsx_lxvp(x * 32, (__vector_pair*)vec_A);
            vec_A1 = __builtin_vsx_lxvp(128 + (x * 32), (__vector_pair*)vec_A);
            /*printf("x=%d\n", x);
            vec_t c[2];
            __builtin_vsx_disassemble_pair(c, &a_pair_bot);
            dump_vec_bf16("c low", c[0]);
            dump_vec_bf16("c high", c[1]);*/
            // Accumulate into the 8 blocks
             mma_plus_instr<T>::outer_product(&acc[0], &vec_A0, vec_B[2*x]);
             mma_plus_instr<T>::outer_product(&acc[1], &vec_A0, vec_B[2*x+1]);
             mma_plus_instr<T>::outer_product(&acc[2], &vec_A0, vec_B[2*x+8]);
             mma_plus_instr<T>::outer_product(&acc[3], &vec_A0, vec_B[2*x+9]);
            
             mma_plus_instr<T>::outer_product(&acc[4], &vec_A1, vec_B[2*x]);
             mma_plus_instr<T>::outer_product(&acc[5], &vec_A1, vec_B[2*x+1]);
             mma_plus_instr<T>::outer_product(&acc[6], &vec_A1, vec_B[2*x+8]);
             mma_plus_instr<T>::outer_product(&acc[7], &vec_A1, vec_B[2*x+9]);
        }
    }
    // ... Save accumulators logic ...
     // Top Half (Rows ii to ii+7)
        save_dmr1024(&acc[0], ii, jj);
        save_dmr1024(&acc[1], ii, jj + 4);
        save_dmr1024(&acc[2], ii, jj + 8);
        save_dmr1024(&acc[3], ii, jj + 12);

        // Bottom Half (Rows ii+8 to ii+15)
        save_dmr1024(&acc[4], ii + 8, jj);
        save_dmr1024(&acc[5], ii + 8, jj + 4);
        save_dmr1024(&acc[6], ii + 8, jj + 8);
        save_dmr1024(&acc[7], ii + 8, jj + 12);
}
        
     void KERNEL_16x4(int64_t ii, int64_t jj) {
        printf("In MMA+ kernel 16x4\n");
        vec_t vec_A[16], vec_B[4], vec_C[8];
        __dmr1024 acc_0, acc_1;
        __vector_pair vec_A0, vec_A1;

        __builtin_mma_dmsetdmrz(&acc_0);
        __builtin_mma_dmsetdmrz(&acc_1);

        for (int l = 0; l < k; l+=8) {
            packNormal<T>(A+(ii*lda)+l, lda, 16, 8, (uint8_t*)vec_A);
            /*for (int i = 0; i < 4; i++) {
                vec_A[i*2]     = temp_A[i];
                vec_A[i*2 + 1] = temp_A[i + 4];

                vec_A[i*2 + 8] = temp_A[i + 8];
                vec_A[i*2 + 9] = temp_A[i + 12];
            }*/
            packNormal<T>(B+(jj*ldb)+l, ldb, 8, 4, (uint8_t*)vec_B);
            for (int x = 0; x < 4; x++) {
                vec_A0 = __builtin_vsx_lxvp(x * 32, (__vector_pair*)vec_A);
                vec_A1 = __builtin_vsx_lxvp(128 + (x * 32), (__vector_pair*)vec_A);
                
                // __builtin_vsx_build_pair(&vec_A0,vec_A[x], vec_A[x+4]);
                //__builtin_vsx_build_pair(&vec_A1,vec_A[x+8], vec_A[x+12]);

                mma_plus_instr<T>::outer_product(&acc_0, &vec_A0, vec_B[x]);
                mma_plus_instr<T>::outer_product(&acc_1, &vec_A1, vec_B[x]);
            }
        }

        save_dmr1024(&acc_0, ii,     jj);
        save_dmr1024(&acc_1, ii+8,    jj);
    }

    void KERNEL_16x8(int64_t ii, int64_t jj) {
	  vec_t vec_A[16], vec_B[8], vec_C[8];
        __dmr1024 acc_0, acc_1, acc_2, acc_3;
	__vector_pair vec_A0, vec_A1;
        __builtin_mma_dmsetdmrz(&acc_0);
        __builtin_mma_dmsetdmrz(&acc_1);
        __builtin_mma_dmsetdmrz(&acc_2);
        __builtin_mma_dmsetdmrz(&acc_3);
        for (int l = 0; l < k; l+=8) {
            packNormal<T>(A+(ii*lda)+l, lda, 16, 8, (uint8_t*)vec_A);
            packNormal<T>(B+(jj*ldb)+l, ldb, 8, 8, (uint8_t*)vec_B);
            for (int x = 0; x < 4; x++) {
                vec_A0 = __builtin_vsx_lxvp(x * 32, (__vector_pair*)vec_A);
                vec_A1 = __builtin_vsx_lxvp(128 + (x * 32), (__vector_pair*)vec_A);
		    //__builtin_vsx_build_pair(&vec_A0,vec_A[x], vec_A[x+4]);
		    //__builtin_vsx_build_pair(&vec_A1,vec_A[x+8], vec_A[x+12]);
	 	 mma_plus_instr<T>::outer_product(&acc_0, &vec_A0, vec_B[2*x]);
                 mma_plus_instr<T>::outer_product(&acc_1, &vec_A1, vec_B[2*x]);
                 mma_plus_instr<T>::outer_product(&acc_2, &vec_A0, vec_B[2*x + 1]);
                 mma_plus_instr<T>::outer_product(&acc_3, &vec_A1, vec_B[2*x + 1]);

            }
        }
        save_dmr1024(&acc_0, ii,     jj);
        save_dmr1024(&acc_1, ii + 8, jj);
        save_dmr1024(&acc_2, ii,     jj + 4);
        save_dmr1024(&acc_3, ii + 8, jj + 4);

    }
    void KERNEL_8x16(int64_t ii, int64_t jj) {
	  vec_t vec_A[8], vec_B[16], vec_C[8];
        __dmr1024 acc_0, acc_1, acc_2, acc_3;
	__vector_pair vec_A0;// vec_A1;
        __builtin_mma_dmsetdmrz(&acc_0);
        __builtin_mma_dmsetdmrz(&acc_1);
        __builtin_mma_dmsetdmrz(&acc_2);
        __builtin_mma_dmsetdmrz(&acc_3);
        for (int l = 0; l < k; l+=8) {
            packNormal<T>(A+(ii*lda)+l, lda, 8, 8, (uint8_t*)vec_A);
            packNormal<T>(B+(jj*ldb)+l, ldb, 8, 16, (uint8_t*)vec_B);
            for (int x = 0; x < 4; x++) {
		    //__builtin_vsx_build_pair(&vec_A0,vec_A[x], vec_A[x+4]);
                vec_A0 = __builtin_vsx_lxvp(x * 32, (__vector_pair*)vec_A);

                mma_plus_instr<T>::outer_product(&acc_0, &vec_A0, vec_B[2*x]);
                 mma_plus_instr<T>::outer_product(&acc_1, &vec_A0, vec_B[2*x + 1]);
                 mma_plus_instr<T>::outer_product(&acc_2, &vec_A0, vec_B[2*x + 8]);
                 mma_plus_instr<T>::outer_product(&acc_3, &vec_A0, vec_B[2*x + 9]);
            }
        }

        save_dmr1024(&acc_0, ii,     jj);
        save_dmr1024(&acc_1, ii,     jj + 4);
        save_dmr1024(&acc_2, ii,     jj + 8);
        save_dmr1024(&acc_3, ii,     jj + 12);
    }
    template<int RM, int RN>
    void gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        mma_common::gemm_small_impl<T, RM, RN>(
            m0, m, n0, n, A, lda, B, ldb, C, ldc, k, ith, nth,
            mma_instr<T>::outer_product
        );
    }

    template<int RM>
    void gemm_Mx8(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        mma_common::gemm_Mx8_impl<T, RM>(
            m0, m, n0, n, A, lda, B, ldb, C, ldc, k, ith, nth,
            mma_instr<T>::outer_product
        );
    }

    template<int RM, int RN>
    inline void kernel(int64_t ii, int64_t jj) {
       if constexpr(RM == 16 && RN == 16) {
          KERNEL_16x16(ii,jj);
       } else if constexpr(RM == 16 && RN == 8) {
          KERNEL_16x8(ii,jj);
        } else if constexpr(RM == 8 && RN == 16) {
          KERNEL_8x16(ii,jj);
        } else if constexpr(RM == 16 && RN == 4) {
          KERNEL_16x4(ii, jj);
        } else if constexpr(RM == 4 && RN == 8) {
          KERNEL_4x8(ii,jj);
       } else if constexpr(RM == 8 && RN == 8) {
          KERNEL_8x8(ii,jj);
       } else if constexpr(RM == 8 && RN == 4) {
          KERNEL_8x4(ii,jj);
       } else {
          assert(false && "RN/RM values not supported");
       }
    }

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
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
            kernel<RM, RN>(ii, jj);
        }
    }

    const T *const A;
    const T *const B;
    float *C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};
// --- BOTTOM OF sgemm-p12.cpp ---

extern "C" {
    void run_sgemm_p12_bf16(int64_t k, 
                            const void * A, int64_t lda, 
                            const void * B, int64_t ldb, 
                            float * C, int64_t ldc, 
                            int ith, int nth,
                            int64_t m, int64_t n) {
        
        // This file knows what tinyBLAS_PPC_P12 is, 
        // so it can instantiate it here.
        tinyBLAS_HP16_P12<ggml_bf16_t> tb{
            k, 
            (const ggml_bf16_t *)A, lda, 
            (const ggml_bf16_t *)B, ldb, 
            (float* )C, ldc, 
            ith, nth
        };

        tb.matmul(m, n);
    }
    void run_sgemm_p12_fp16(int64_t k, 
                            const void * A, int64_t lda, 
                            const void * B, int64_t ldb, 
                            float * C, int64_t ldc, 
                            int ith, int nth,
                            int64_t m, int64_t n) {
        
        // This file knows what tinyBLAS_PPC_P12 is, 
        // so it can instantiate it here.
        tinyBLAS_HP16_P12<ggml_fp16_t> tb{
            k, 
            (const ggml_fp16_t *)A, lda, 
            (const ggml_fp16_t *)B, ldb, 
            (float* )C, ldc, 
            ith, nth
        };

        tb.matmul(m, n);
    }
}
#endif
