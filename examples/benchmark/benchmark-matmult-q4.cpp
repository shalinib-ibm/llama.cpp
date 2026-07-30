#include "common.h"
#include "ggml.h"
#include "build-info.h"
#include "../../ggml/src/ggml-common.h"
#include <locale.h>
#include <assert.h>
#include <math.h>
#include <cstring>
#include <cstdio>
#include <cinttypes>
#include <unordered_map>
#include <queue>
#include <string.h>
#include <cassert>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static float tensor_sum_elements(const ggml_tensor * tensor) {
    double sum1;
    if (tensor->type == GGML_TYPE_F32) {
        for (int j = 0; j < tensor->ne[1]; j++) {
            for (int k = 0; k < tensor->ne[0]; k++) {
		    sum1 = sum1 + ((float *) tensor->data)[j*tensor->ne[0] + k];
            }
	    printf("\n");
        }
    }
    return sum1;
}

static void tensor_dump(const ggml_tensor * tensor, const char * name) {
    printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi) - \n", name,
        tensor->type, ggml_type_name(tensor->type),
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[0], tensor->nb[1], tensor->nb[2]);
    float sum = tensor_sum_elements(tensor);
    printf("Sum of tensor %s is %6.2f\n", name, sum);
    printf("tensor data\n");
    int rows = tensor->ne[1];
    int cols = tensor->ne[0];
    float *data = (float*)tensor->data;
    for (int r = 0; r < rows; r++) {
	for (int c = 0; c < cols; c++) {
		printf("%.2f ", data[r*cols+c]);
	}
	printf("\n");
    }
}

/*static void tensor_dump_dequant(const ggml_tensor * tensor, const char * name) {
    int rows = tensor->ne[1];
    int cols = tensor->ne[0];

    // allocate temp buffer
    std::vector<float> tmp(ggml_nelements(tensor));
    ggml_get_data_f32(tensor, tmp.data());

    printf("%15s (dequantized): type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 "\n",
           name, tensor->type, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1]);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%.2f ", tmp[r*cols + c]);
        }
        printf("\n");
    }
}*/

#define TENSOR_DUMP(tensor) tensor_dump(tensor, #tensor)

struct benchmark_params_struct {
    int     n_threads     = 1;
    int32_t n_iterations  = 10;
};

static void print_usage(int /*argc*/, char ** argv, struct benchmark_params_struct params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -i N, --iter N     number of iterations to use during computation (default: %d)\n", params.n_iterations);
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv)  {
    struct benchmark_params_struct benchmark_params;

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-i" || arg == "--iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_iterations = std::stoi(argv[i]);
        }  else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, benchmark_params);
            exit(0);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv, benchmark_params);
        exit(1);
    }

    llama_print_build_info();
    printf("Starting Test\n");

    // create the ggml context
    struct ggml_context * ctx;
    //const int sizex = 4096;
    //const int sizey = 11008;

#undef VERBOSE_DEBUGGING
#ifndef VERBOSE_DEBUGGING
    /*
    const int sizey = 4096;
    const int sizex = 11008;
    const int sizez = 128;
    */
    const int sizey = 16;
    const int sizex = 3*32;
    const int sizez = 16;
#else
    /* Working - let's increase size */
    const int sizey = 1;
    const int sizex = (8*32);
    const int sizez = 1;

    /*const int sizey = 1;
    const int sizex = 3*(8*32);
    const int sizez = 1;*/
#endif

    //printf("Memsize required = %i\n", sizex*sizex);

    // TODO: perform the bench for all types or for a user specified type
    const ggml_type qtype1 = GGML_TYPE_Q8_0;
    const ggml_type qtype2 = GGML_TYPE_Q4_0;

    size_t ctx_size = 0;
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey);
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey);
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizez);
    ctx_size += ggml_row_size(qtype1,         sizex*sizey);
    ctx_size += ggml_row_size(qtype1,         sizex*sizez);
    ctx_size += ggml_row_size(qtype1,         sizex*sizey);
    ctx_size += ggml_row_size(qtype2,         sizex*sizey);
    ctx_size += ggml_row_size(qtype2,         sizex*sizez);
    ctx_size += ggml_row_size(qtype2,         sizex*sizey);
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey); // BLAS
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey); // BLAS
    ctx_size += 1024*1024*16;

    printf("Allocating Memory of size %zi bytes, %zi MB\n",ctx_size, (ctx_size/1024/1024));

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /* no_alloc   =*/ 0
    };

    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return 1;
    }


    printf("Creating new tensors\n");
    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m11 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
    //ggml_set_f32(m11, -1.23f);
    float *m11_data = (float *) m11->data;
    float x = 1.0f;
    for (int r = 0; r < m11->ne[1]; r++) {
        for (int c = 0; c < m11->ne[0]; c++) {
            m11_data[r*m11->ne[0] + c] = x++;//(float)(r*m11->ne[0] + c + 1);
        }
     }

    TENSOR_DUMP(m11);
    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m12 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
    ggml_set_f32(m12, 1.5f);

    // printf("Creating new tensor m2\n");
    struct ggml_tensor * m2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizez);
    float *m2_data = (float *) m2->data;
    //ggml_set_f32(m2, -12.23f);
    float y = 2.0f;
    for (int r = 0; r < m2->ne[1]; r++) {
        for (int c = 0; c < m2->ne[0]; c++) {
            m2_data[r*m2->ne[0] + c] = y++;//(float)(r*m11->ne[0] + c + 1);
        }
     }

    printf("\n------ Test 1 - Matrix Mult via F32 code\n");
    // printf("Creating new tensor m11xm2\n");
    struct ggml_tensor * m11xm2 = ggml_mul_mat(ctx, m11, m2);

    // printf("Creating compute graph\n");
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, m11xm2);

    printf("n_threads=%i\n", benchmark_params.n_threads);

    //TENSOR_DUMP(m11);
    //TENSOR_DUMP(m2);

    std::vector<uint8_t> work_buffer;

    ggml_graph_compute_helper(work_buffer, gf, benchmark_params.n_threads);

    //TENSOR_DUMP(ggml_graph_node(gf, 0));

    printf("\n------ Test 2 - Matrix Mult via %s and %s code\n", ggml_type_name(qtype1), ggml_type_name(qtype2));

    int32_t nelements = sizex*sizey;
    int32_t nelements2 = sizex*sizez;

    // Set up a the benchmark matrices
    printf("Creating new tensor of type Q8_0 & Running quantize\n");
    struct ggml_tensor * q1 = ggml_new_tensor_2d(ctx, qtype1, sizex, sizey);
    ggml_quantize_chunk(qtype1, (const float *) m11->data, q1->data, 0, nelements/m11->ne[0], m11->ne[0], nullptr);
    //tensor_dump_dequant(q1, "Q4_0 q1 tensor");

    printf("Creating new tensor of type Q4_0 & Running quantize\n");
    struct ggml_tensor * q2 = ggml_new_tensor_2d(ctx, qtype2, sizex, sizez);
    ggml_quantize_chunk(qtype2, (const float *) m2->data, q2->data, 0, nelements2/m2->ne[0], m2->ne[0], nullptr);
    //TENSOR_QDUMP(q2);
     
    printf("Creating new tensor of type Q8_0 & Running quantize\n");
    struct ggml_tensor * q3 = ggml_new_tensor_2d(ctx, qtype1, sizex, sizez);
    ggml_quantize_chunk(qtype1, (const float *) m2->data, q3->data, 0, nelements2/m2->ne[0], m2->ne[0], nullptr);
    TENSOR_DUMP(q3);
    
    // Set up a the compute graph
    // printf("Creating new tensor q31\n");
    struct ggml_tensor * qprod = ggml_mul_mat(ctx, q2, q1);
    //struct ggml_tensor * qprod = ggml_mul_mat(ctx, q3, q1);

    // printf("Creating compute graph\n");
    struct ggml_cgraph * gf31 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf31, qprod);

    // Set up a second graph computation to make sure we override the CPU cache lines
    // printf("Creating new tensor q12 & Running quantize\n");
    struct ggml_tensor * q12 = ggml_new_tensor_2d(ctx, qtype1, sizex, sizey);
    ggml_quantize_chunk(qtype1, (const float *) m12->data, q12->data, 0, nelements/m12->ne[0], m12->ne[0], nullptr);

    // printf("Creating new tensor q32\n");
    struct ggml_tensor * q32 = ggml_mul_mat(ctx, q12, m2);

    //printf("Creating compute graph\n");
    struct ggml_cgraph * gf32 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf32, q32);
    printf("n_threads=%i\n", benchmark_params.n_threads);

    const int dimx = sizex;
    const int dimy = sizey;
    const int dimz = sizez;
    long long int flops_per_dot_product = dimy + dimy;
    long long int flops_per_matrix = flops_per_dot_product * dimx * dimz; ;
    printf("Matrix Multiplication of (%i,%i,%i) x (%i,%i,%i) - about %6.2f gFLOPS\n\n", sizex, sizey, 1, sizex, sizez, 1, 1.0f*flops_per_matrix / 1000 / 1000 / 1000);


    // Let's use the F32 result from above as a reference for the quantized multiplication
    float sum_of_F32_reference = tensor_sum_elements(ggml_graph_node(gf, 0));

    printf("Iteration;NThreads; SizeX; SizeY; SizeZ; Required_FLOPS; Elapsed_u_Seconds; gigaFLOPS\n");
    printf("=====================================================================================\n");

    double  gflops_sum = 0;
    for (int i=0;i<benchmark_params.n_iterations ;i++) {

        long long int start = ggml_time_us();
        //printf("Running ggml_graph_compute\n");
        ggml_graph_compute_helper(work_buffer, gf31, benchmark_params.n_threads);
        TENSOR_DUMP(ggml_graph_node(gf31, 0));

        long long int stop = ggml_time_us();
        long long int usec = stop-start;
        double gflops = (double)(flops_per_matrix)/usec/1000.0;
        gflops_sum += gflops;
        printf("%9i;%8i;%6i;%6i;%6i;%15lli;%18lli;%10.2f\n",
            i,
            benchmark_params.n_threads,
            sizex, sizey, sizez, flops_per_matrix,
            usec,gflops);

#ifdef VERBOSE_DEBUGGING
        //TENSOR_DUMP("res",gf31.nodes[0])
#endif

        // Check that the matrix multiplication result is in the right ballpark
        // We cannot use the exact value from the F32 multiplication because the quantizuation will be slightly different
        float sum_of_Q4_result = tensor_sum_elements(ggml_graph_node(gf31, 0));
        float delta = std::abs(sum_of_Q4_result - sum_of_F32_reference);
        float allowed_delta = (sum_of_F32_reference) / 1000 / 1000; //  Let's accept an epsilon of 10^-6

        if (delta > allowed_delta)  {
            printf("\nABORT - ERROR in Matrix Multiplication result - expected %6.2f, got %6.2f (delta %6.2f > allowed_delta %6.2f)\n",
                sum_of_F32_reference,
                sum_of_Q4_result,
                delta,
                allowed_delta
            );
            exit(0);
        }

        // Running a different graph computation to make sure we override the CPU cache lines
        ggml_graph_compute_helper(work_buffer, gf32, benchmark_params.n_threads);
    }
    printf("\n");
    printf("Average%78.2f\n",gflops_sum/((double)benchmark_params.n_iterations));
    printf("=====================================================================================\n");
}
