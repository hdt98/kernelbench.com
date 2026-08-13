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



// ===== 3-buffer pipeline + register-blocked (2 N-tiles per WF) =====
#define DB_BK3 64
#define DB_BKP3 68
#define N_BUF3 3

template<int BM, int BN, int NWFX, int NWFY, int BS, bool ATOMIC>
__global__ __launch_bounds__(BS) void gemm_db3_r2(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    void* __restrict__ D, const float* __restrict__ ws,
    int M, int N, int K, int split_k) {
    __shared__ uint8_t As[N_BUF3][BM][DB_BKP3];
    __shared__ uint8_t Bs[N_BUF3][BN][DB_BKP3];
    int tid=threadIdx.x, wf=tid/WF_SIZE, lane=tid%WF_SIZE;
    int wmx=wf%NWFX, wmy=wf/NWFX, m_off=blockIdx.x*BM, n_off=blockIdx.y*BN;
    int mm=lane%32, kg=lane/32;
    __f16 acc0={},acc1={}; float* ap0=(float*)&acc0; float* ap1=(float*)&acc1;
    int kb_tot=(K+DB_BK3-1)/DB_BK3, kbps=(kb_tot+split_k-1)/split_k;
    int ks=blockIdx.z*kbps, ke=ks+kbps; if(ke>kb_tot) ke=kb_tot;
    int ni=ke-ks; if(ni<=0)return;
    const int ta=BM*DB_BK3/16, tt=(BM+BN)*DB_BK3/16, lpt=(tt+BS-1)/BS;

    // Prologue: load 2 K-blocks (buf[0] and buf[1])
    for(int blk=0; blk<2 && blk<ni; blk++){
        int kb_base=(ks+blk)*DB_BK3;
        for(int i=0;i<lpt;i++){
            int idx=i*BS+tid;
            if(idx<ta){int r=idx/(DB_BK3/16),c16=idx%(DB_BK3/16),c=c16*16;int gr=m_off+r,gc=kb_base+c;uint4 v={0,0,0,0};
            if(gr<M&&gc+15<K){v=*(const uint4*)&A[gr*K+gc];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
            else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gc+j;if(gr<M&&g<K){uint8_t b=A[gr*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&As[blk][r][c]=v;}
            else if(idx<tt){int bi=idx-ta,r=bi/(DB_BK3/16),c16=bi%(DB_BK3/16),c=c16*16;int gn=n_off+r,gk=kb_base+c;uint4 v={0,0,0,0};
            if(gn<N&&gk+15<K){v=*(const uint4*)&B[gn*K+gk];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
            else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gk+j;if(gn<N&&g<K){uint8_t b=B[gn*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&Bs[blk][r][c]=v;}
        }
        __syncthreads();
    }

    // Pipeline: compute buf[it%3], load buf[(it+2)%3] (2 blocks ahead)
    for(int it=0;it<ni-2;it++){
        int kb=ks+it, cb=it%N_BUF3, nb=(it+2)%N_BUF3, nb_base=(kb+2)*DB_BK3;
        // Compute (2 N-tiles, A shared)
        #pragma unroll
        for(int kc=0;kc<DB_BK3/16;kc++){
            int ko=kc*16+kg*8;
            int64_t av=0;uint8_t*ab=(uint8_t*)&av;int ar=wmy*32+mm;
            *(uint32_t*)ab=*(uint32_t*)&As[cb][ar][ko];*(uint32_t*)(ab+4)=*(uint32_t*)&As[cb][ar][ko+4];
            int bc0=wmx*64+mm,bc1=wmx*64+32+mm;
            int64_t bv0=0;uint8_t*bb0=(uint8_t*)&bv0; *(uint32_t*)bb0=*(uint32_t*)&Bs[cb][bc0][ko];*(uint32_t*)(bb0+4)=*(uint32_t*)&Bs[cb][bc0][ko+4];
            int64_t bv1=0;uint8_t*bb1=(uint8_t*)&bv1; *(uint32_t*)bb1=*(uint32_t*)&Bs[cb][bc1][ko];*(uint32_t*)(bb1+4)=*(uint32_t*)&Bs[cb][bc1][ko+4];
            acc0=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv0,acc0,0,0,0);
            acc1=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv1,acc1,0,0,0);
        }
        // Load next (2 blocks ahead)
        for(int i=0;i<lpt;i++){
            int idx=i*BS+tid;
            if(idx<ta){int r=idx/(DB_BK3/16),c16=idx%(DB_BK3/16),c=c16*16;int gr=m_off+r,gc=nb_base+c;uint4 v={0,0,0,0};
            if(gr<M&&gc+15<K){v=*(const uint4*)&A[gr*K+gc];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
            else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gc+j;if(gr<M&&g<K){uint8_t b=A[gr*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&As[nb][r][c]=v;}
            else if(idx<tt){int bi=idx-ta,r=bi/(DB_BK3/16),c16=bi%(DB_BK3/16),c=c16*16;int gn=n_off+r,gk=nb_base+c;uint4 v={0,0,0,0};
            if(gn<N&&gk+15<K){v=*(const uint4*)&B[gn*K+gk];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
            else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gk+j;if(gn<N&&g<K){uint8_t b=B[gn*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&Bs[nb][r][c]=v;}
        }
        __syncthreads();
    }

    // Epilogue: compute remaining 2 K-blocks
    for(int rem=0;rem<2 && (ni-2+rem)>=0;rem++){
        int lb=(ni-2+rem)%N_BUF3; if(ni-2+rem<0) break;
        #pragma unroll
        for(int kc=0;kc<DB_BK3/16;kc++){
            int ko=kc*16+kg*8;
            int64_t av=0;uint8_t*ab=(uint8_t*)&av;int ar=wmy*32+mm;
            *(uint32_t*)ab=*(uint32_t*)&As[lb][ar][ko];*(uint32_t*)(ab+4)=*(uint32_t*)&As[lb][ar][ko+4];
            int bc0=wmx*64+mm,bc1=wmx*64+32+mm;
            int64_t bv0=0;uint8_t*bb0=(uint8_t*)&bv0; *(uint32_t*)bb0=*(uint32_t*)&Bs[lb][bc0][ko];*(uint32_t*)(bb0+4)=*(uint32_t*)&Bs[lb][bc0][ko+4];
            int64_t bv1=0;uint8_t*bb1=(uint8_t*)&bv1; *(uint32_t*)bb1=*(uint32_t*)&Bs[lb][bc1][ko];*(uint32_t*)(bb1+4)=*(uint32_t*)&Bs[lb][bc1][ko+4];
            acc0=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv0,acc0,0,0,0);
            acc1=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv1,acc1,0,0,0);
        }
    }

    int col0=n_off+wmx*64+mm,col1=n_off+wmx*64+32+mm;
    if(ATOMIC){float*Df=(float*)D;for(int e=0;e<16;e++){int row=m_off+wmy*32+(e%4)+(e/4)*8+kg*4;
        if(row<M&&col0<N)atomicAdd(&Df[row*N+col0],ap0[e]*4.0f);if(row<M&&col1<N)atomicAdd(&Df[row*N+col1],ap1[e]*4.0f);}}
    else{uint16_t*Db=(uint16_t*)D;float s0=(col0<N)?ws[col0]:0.0f,s1=(col1<N)?ws[col1]:0.0f;
    for(int e=0;e<16;e++){int row=m_off+wmy*32+(e%4)+(e/4)*8+kg*4;
        if(row<M&&col0<N)Db[row*N+col0]=f32_to_bf16_bits(ap0[e]*4.0f*s0);if(row<M&&col1<N)Db[row*N+col1]=f32_to_bf16_bits(ap1[e]*4.0f*s1);}}
}


extern "C" {
int mfma_fp8_gemm_db3(const void* a_ptr, const void* b_ptr, void* d_ptr,
    int M, int N, int K, int split_k, const void* weight_scale, void* stream) {
    hipStream_t hip_stream = (hipStream_t)stream;
    const float* ws = (const float*)weight_scale;
    dim3 grid((M+127)/128, (N+127)/128, split_k);
    if (split_k == 1)
        gemm_db3_r2<128,128,2,4,512,false><<<grid,512,0,hip_stream>>>((const uint8_t*)a_ptr,(const uint8_t*)b_ptr,d_ptr,ws,M,N,K,split_k);
    else
        gemm_db3_r2<128,128,2,4,512,true><<<grid,512,0,hip_stream>>>((const uint8_t*)a_ptr,(const uint8_t*)b_ptr,d_ptr,ws,M,N,K,split_k);
    return 0;
}
}
