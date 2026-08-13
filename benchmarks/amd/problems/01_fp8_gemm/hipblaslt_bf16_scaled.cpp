#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstddef>
typedef void* hipblasLtHandle_t;
typedef void* hipblasLtMatmulDesc_t;
typedef void* hipblasLtMatrixLayout_t;
typedef void* hipblasLtMatmulPreference_t;
typedef int hipblasStatus_t;
typedef int hipblasComputeType_t;
#define HB_COMPUTE_32F 2
#define HB_R_32F 0
#define HB_R_16BF 14
#define HB_DESC_TRANSA 0
#define HB_DESC_TRANSB 1
#define HB_PREF_MAX_WS 1
#define HB_OP_N 111
#define HB_OP_T 112
struct AlgoBuf { char data[4096]; };
struct HeuristicResult { AlgoBuf algo; size_t workspaceSize; int state; float wavesCount; int reserved[4]; };
extern "C" {
    hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t*);
    hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t*, hipblasComputeType_t, hipDataType);
    hipblasStatus_t hipblasLtMatmulDescDestroy(hipblasLtMatmulDesc_t);
    hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t, int, const void*, size_t);
    hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t*, hipDataType, uint64_t, uint64_t, int64_t);
    hipblasStatus_t hipblasLtMatrixLayoutDestroy(hipblasLtMatrixLayout_t);
    hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t*);
    hipblasStatus_t hipblasLtMatmulPreferenceDestroy(hipblasLtMatmulPreference_t);
    hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t, int, const void*, size_t);
    hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t, hipblasLtMatmulDesc_t,
        hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
        hipblasLtMatmulPreference_t, int, void*, int*);
    hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t, hipblasLtMatmulDesc_t,
        const void*, const void*, hipblasLtMatrixLayout_t,
        const void*, hipblasLtMatrixLayout_t,
        const void*, const void*, hipblasLtMatrixLayout_t,
        void*, hipblasLtMatrixLayout_t,
        const void*, void*, size_t, hipStream_t);
}
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float val) {
    uint32_t b = __float_as_uint(val);
    uint16_t r = (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
    if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u) r = 0x7FC0u;
    return r;
}
__global__ void scale_cast_kernel(const float* __restrict__ in, uint16_t* __restrict__ out,
                                   const float* __restrict__ scale, int total, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) { out[idx] = f32_to_bf16_bits(in[idx] * scale[idx % N]); }
}

static hipblasLtHandle_t g_handle = nullptr;
static void* g_workspace = nullptr;
static size_t g_ws_sz = 32*1024*1024;
static void* g_f32_out = nullptr;
static int g_f32_out_sz = 0;

// Cache: descriptors and algorithm keyed by (M, N, K)
static int g_cache_m = 0, g_cache_n = 0, g_cache_k = 0;
static hipblasLtMatmulDesc_t g_desc = nullptr;
static hipblasLtMatrixLayout_t g_ad = nullptr, g_bd = nullptr, g_dd = nullptr;
static AlgoBuf g_algo;

static void setup_cache(int M, int N, int K) {
    if (g_desc && g_cache_m == M && g_cache_n == N && g_cache_k == K) return;
    if (g_desc) {
        hipblasLtMatrixLayoutDestroy(g_ad); hipblasLtMatrixLayoutDestroy(g_bd); hipblasLtMatrixLayoutDestroy(g_dd);
        hipblasLtMatmulDescDestroy(g_desc);
    }
    hipblasLtMatmulDescCreate(&g_desc, (hipblasComputeType_t)HB_COMPUTE_32F, (hipDataType)HB_R_32F);
    int32_t tA=HB_OP_T, tB=HB_OP_N;
    hipblasLtMatmulDescSetAttribute(g_desc, HB_DESC_TRANSA, &tA, sizeof(int32_t));
    hipblasLtMatmulDescSetAttribute(g_desc, HB_DESC_TRANSB, &tB, sizeof(int32_t));
    hipblasLtMatrixLayoutCreate(&g_ad, (hipDataType)HB_R_16BF, (uint64_t)K, (uint64_t)N, (int64_t)K);
    hipblasLtMatrixLayoutCreate(&g_bd, (hipDataType)HB_R_16BF, (uint64_t)K, (uint64_t)M, (int64_t)K);
    hipblasLtMatrixLayoutCreate(&g_dd, (hipDataType)HB_R_32F, (uint64_t)N, (uint64_t)M, (int64_t)N);
    // Get algorithm
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref, HB_PREF_MAX_WS, &g_ws_sz, sizeof(size_t));
    HeuristicResult heur; int n=0;
    hipblasLtMatmulAlgoGetHeuristic(g_handle, g_desc, g_ad, g_bd, g_dd, g_dd, pref, 1, &heur, &n);
    g_algo = heur.algo;
    hipblasLtMatmulPreferenceDestroy(pref);
    g_cache_m = M; g_cache_n = N; g_cache_k = K;
}

extern "C" int hipblaslt_bf16_gemm_scaled(
    const void* a_ptr, const void* b_ptr, uint16_t* d_ptr,
    int M, int N, int K, const float* scale_ptr, hipStream_t stream
) {
    if (!g_handle) { hipblasLtCreate(&g_handle); hipMalloc(&g_workspace, g_ws_sz); }
    int out_sz = M*N*(int)sizeof(float);
    if (g_f32_out_sz < out_sz) { if (g_f32_out) hipFree(g_f32_out); hipMalloc(&g_f32_out, out_sz); g_f32_out_sz = out_sz; }
    setup_cache(M, N, K);
    float alpha=1.0f, beta=0.0f;
    hipblasStatus_t st = hipblasLtMatmul(g_handle, g_desc, &alpha, b_ptr, g_ad, a_ptr, g_bd,
        &beta, g_f32_out, g_dd, g_f32_out, g_dd, &g_algo, g_workspace, g_ws_sz, stream);
    if (st!=0) return -4;
    int total=M*N, blk=256;
    hipLaunchKernelGGL(scale_cast_kernel, dim3((total+blk-1)/blk), dim3(blk), 0, stream,
                        (const float*)g_f32_out, d_ptr, scale_ptr, total, N);
    return 0;
}
