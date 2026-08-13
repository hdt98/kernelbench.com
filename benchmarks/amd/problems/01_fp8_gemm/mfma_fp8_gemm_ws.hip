// FP8 e4m3 GEMM via raw MFMA intrinsics for AMD MI325X (gfx942, CDNA 3).
// v11: Hybrid kernel — single-buffer BK=128 for rectangular shapes,
//     double-buffer BK=64 for square shapes (memory-compute overlap).
//
// gfx942 FNUZ fp8: 4.0x correction. 0x80 bytes sanitized during load.
// MFMA output: acc[e] = C[(e%4)+(e/4)*8+k_group*4][lane%32]

#include <hip/hip_runtime.h>
#include <cstdint>

#define WF_SIZE 64
typedef float __f16 __attribute__((ext_vector_type(16)));

__device__ __forceinline__ uint32_t sanitize_fp8_u32(uint32_t val) {
    uint32_t n = ((val & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) & 0x80808080u;
    uint32_t az = ~n & 0x80808080u;
    uint32_t ms = val & 0x80808080u;
    return val & ~(az & ms);
}
__device__ __forceinline__ uint64_t sanitize_fp8_u64(uint64_t val) {
    uint64_t mask7 = 0x7F7F7F7F7F7F7F7Full;
    uint64_t mask8 = 0x8080808080808080ull;
    uint64_t n = ((val & mask7) + mask7) & mask8;
    uint64_t az = ~n & mask8;
    uint64_t ms = val & mask8;
    return val & ~(az & ms);
}

__device__ __forceinline__ uint4 sanitize_fp8_u4(uint4 v) {
    v.x = sanitize_fp8_u32(v.x);
    v.y = sanitize_fp8_u32(v.y);
    v.z = sanitize_fp8_u32(v.z);
    v.w = sanitize_fp8_u32(v.w);
    return v;
}
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float val) {
    uint32_t b = __float_as_uint(val);
    uint16_t r = (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
    if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u) r = 0x7FC0u;
    return r;
}

// ===== Single-buffer kernel (BK=128) =====
#define SB_BK 128
#define SB_BKP 132

#define DB_BK 64
#define DB_BKP 68
#define N_BUF 2



// Warp-specialized: 8 producer WFs + 8 consumer WFs
template<int BM, int BN, int BS, bool ATOMIC>
__global__ __launch_bounds__(BS) void gemm_ws(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    void* __restrict__ D, const float* __restrict__ ws,
    int M, int N, int K, int split_k) {
    __shared__ uint8_t As[N_BUF][BM][DB_BKP];
    __shared__ uint8_t Bs[N_BUF][BN][DB_BKP];
    int tid=threadIdx.x, wf=tid/WF_SIZE, lane=tid%WF_SIZE;
    bool is_prod = wf < 8;
    int mm=lane%32, kg=lane/32;
    int kb_tot=(K+DB_BK-1)/DB_BK, kbps=(kb_tot+split_k-1)/split_k;
    int ks=blockIdx.z*kbps, ke=ks+kbps; if(ke>kb_tot) ke=kb_tot;
    int ni=ke-ks; if(ni<=0)return;
    int cwf=wf-8, wmx=cwf%4, wmy=cwf/4;
    int m_off=blockIdx.x*BM, n_off=blockIdx.y*BN;
    __f16 acc={}; float* ap=(float*)&acc;
    const int ta=BM*DB_BK/16, tt=(BM+BN)*DB_BK/16, lpt_p=(tt+511)/512;
    if(is_prod){int kb_base=ks*DB_BK;
    for(int i=0;i<lpt_p;i++){int idx=i*512+tid;
        if(idx<ta){int r=idx/(DB_BK/16),c16=idx%(DB_BK/16),c=c16*16;int gr=m_off+r,gc=kb_base+c;uint4 v={0,0,0,0};
        if(gr<M&&gc+15<K){v=*(const uint4*)&A[gr*K+gc];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
        else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gc+j;if(gr<M&&g<K){uint8_t b=A[gr*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&As[0][r][c]=v;}
        else if(idx<tt){int bi=idx-ta,r=bi/(DB_BK/16),c16=bi%(DB_BK/16),c=c16*16;int gn=n_off+r,gk=kb_base+c;uint4 v={0,0,0,0};
        if(gn<N&&gk+15<K){v=*(const uint4*)&B[gn*K+gk];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
        else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gk+j;if(gn<N&&g<K){uint8_t b=B[gn*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&Bs[0][r][c]=v;}
    }}
    __syncthreads();
    for(int it=0;it<ni-1;it++){
        int kb=ks+it, cb=it%N_BUF, nb=(it+1)%N_BUF, nb_base=(kb+1)*DB_BK;
        if(!is_prod){
#pragma unroll
        for(int kc=0;kc<DB_BK/16;kc++){int ko=kc*16+kg*8;
            int64_t av=0;uint8_t*ab=(uint8_t*)&av;int ar=wmy*32+mm;
            *(uint32_t*)ab=*(uint32_t*)&As[cb][ar][ko];*(uint32_t*)(ab+4)=*(uint32_t*)&As[cb][ar][ko+4];
            int64_t bv=0;uint8_t*bb=(uint8_t*)&bv;int bc=wmx*32+mm;
            *(uint32_t*)bb=*(uint32_t*)&Bs[cb][bc][ko];*(uint32_t*)(bb+4)=*(uint32_t*)&Bs[cb][bc][ko+4];
            acc=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv,acc,0,0,0);}}
        if(is_prod){
        for(int i=0;i<lpt_p;i++){int idx=i*512+tid;
            if(idx<ta){int r=idx/(DB_BK/16),c16=idx%(DB_BK/16),c=c16*16;int gr=m_off+r,gc=nb_base+c;uint4 v={0,0,0,0};
            if(gr<M&&gc+15<K){v=*(const uint4*)&A[gr*K+gc];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
            else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gc+j;if(gr<M&&g<K){uint8_t b=A[gr*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&As[nb][r][c]=v;}
            else if(idx<tt){int bi=idx-ta,r=bi/(DB_BK/16),c16=bi%(DB_BK/16),c=c16*16;int gn=n_off+r,gk=nb_base+c;uint4 v={0,0,0,0};
            if(gn<N&&gk+15<K){v=*(const uint4*)&B[gn*K+gk];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
            else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gk+j;if(gn<N&&g<K){uint8_t b=B[gn*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&Bs[nb][r][c]=v;}
        }}
        __syncthreads();
    }
    if(!is_prod){int lb=(ni-1)%N_BUF;
    #pragma unroll
    for(int kc=0;kc<DB_BK/16;kc++){int ko=kc*16+kg*8;
        int64_t av=0;uint8_t*ab=(uint8_t*)&av;int ar=wmy*32+mm;
        *(uint32_t*)ab=*(uint32_t*)&As[lb][ar][ko];*(uint32_t*)(ab+4)=*(uint32_t*)&As[lb][ar][ko+4];
        int64_t bv=0;uint8_t*bb=(uint8_t*)&bv;int bc=wmx*32+mm;
        *(uint32_t*)bb=*(uint32_t*)&Bs[lb][bc][ko];*(uint32_t*)(bb+4)=*(uint32_t*)&Bs[lb][bc][ko+4];
        acc=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv,acc,0,0,0);}
    int col=n_off+wmx*32+mm;
    if(ATOMIC){float*Df=(float*)D;for(int e=0;e<16;e++){int row=m_off+wmy*32+(e%4)+(e/4)*8+kg*4;if(row<M&&col<N)atomicAdd(&Df[row*N+col],ap[e]*4.0f);}}
    else{uint16_t*Db=(uint16_t*)D;float s=(col<N)?ws[col]:0.0f;for(int e=0;e<16;e++){int row=m_off+wmy*32+(e%4)+(e/4)*8+kg*4;if(row<M&&col<N)Db[row*N+col]=f32_to_bf16_bits(ap[e]*4.0f*s);}}
    }
}

extern "C" {
int mfma_fp8_gemm_ws(const void* a_ptr, const void* b_ptr, void* d_ptr,
    int M, int N, int K, int split_k, const void* weight_scale, void* stream) {
    hipStream_t hip_stream = (hipStream_t)stream;
    const float* ws = (const float*)weight_scale;
    dim3 grid((M+63)/64, (N+127)/128, split_k);
    if (split_k == 1)
        gemm_ws<64,128,1024,false><<<grid,1024,0,hip_stream>>>((const uint8_t*)a_ptr,(const uint8_t*)b_ptr,d_ptr,ws,M,N,K,split_k);
    else
        gemm_ws<64,128,1024,true><<<grid,1024,0,hip_stream>>>((const uint8_t*)a_ptr,(const uint8_t*)b_ptr,d_ptr,ws,M,N,K,split_k);
    return 0;
}
}

