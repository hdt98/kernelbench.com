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

template<int BM, int BN, int NWFX, int NWFY, int BS, bool ATOMIC>
__global__ __launch_bounds__(BS) void gemm_sb(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    void* __restrict__ D, const float* __restrict__ ws,
    int M, int N, int K, int split_k) {
    __shared__ uint8_t As[BM][SB_BKP];
    __shared__ uint8_t Bs[BN][SB_BKP];
    int tid=threadIdx.x, wf=tid/WF_SIZE, lane=tid%WF_SIZE;
    int wmx=wf%NWFX, wmy=wf/NWFX, m_off=blockIdx.x*BM, n_off=blockIdx.y*BN;
    int mm=lane%32, kg=lane/32;
    __f16 acc={}; float* ap=(float*)&acc;
    int kb_tot=(K+SB_BK-1)/SB_BK, kbps=(kb_tot+split_k-1)/split_k;
    int ks=blockIdx.z*kbps, ke=ks+kbps; if(ke>kb_tot) ke=kb_tot;
    const int ta=BM*SB_BK/16, tt=(BM+BN)*SB_BK/16, lpt=(tt+BS-1)/BS;
    for(int kb=ks;kb<ke;kb++){
        int kb_base=kb*SB_BK;
        for(int i=0;i<lpt;i++){
            int idx=i*BS+tid;
            if(idx<ta){
                int r=idx/(SB_BK/16),c16=idx%(SB_BK/16),c=c16*16;
                int gr=m_off+r,gc=kb_base+c; uint4 v={0,0,0,0};
                if(gr<M&&gc+15<K) v=sanitize_fp8_u4(*(const uint4*)&A[gr*K+gc]);
                else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gc+j;if(gr<M&&g<K){uint8_t b=A[gr*K+g];p[j]=(b==0x80u)?0u:b;}}}
                *(uint4*)&As[r][c]=v;
            }else if(idx<tt){
                int bi=idx-ta,r=bi/(SB_BK/16),c16=bi%(SB_BK/16),c=c16*16;
                int gn=n_off+r,gk=kb_base+c; uint4 v={0,0,0,0};
                if(gn<N&&gk+15<K) v=sanitize_fp8_u4(*(const uint4*)&B[gn*K+gk]);
                else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gk+j;if(gn<N&&g<K){uint8_t b=B[gn*K+g];p[j]=(b==0x80u)?0u:b;}}}
                *(uint4*)&Bs[r][c]=v;
            }
        }
        __syncthreads();
        #pragma unroll
        for(int kc=0;kc<SB_BK/16;kc++){
            int ko=kc*16+kg*8;
            int64_t av=0; uint8_t*ab=(uint8_t*)&av; int ar=wmy*32+mm;
            *(uint32_t*)ab=*(uint32_t*)&As[ar][ko]; *(uint32_t*)(ab+4)=*(uint32_t*)&As[ar][ko+4];
            int64_t bv=0; uint8_t*bb=(uint8_t*)&bv; int bc=wmx*32+mm;
            *(uint32_t*)bb=*(uint32_t*)&Bs[bc][ko]; *(uint32_t*)(bb+4)=*(uint32_t*)&Bs[bc][ko+4];
            acc=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv,acc,0,0,0);
        }
        __syncthreads();
    }
    int col=n_off+wmx*32+mm;
    if(ATOMIC){
        float*Df=(float*)D;
        for(int e=0;e<16;e++){int row=m_off+wmy*32+(e%4)+(e/4)*8+kg*4;if(row<M&&col<N)atomicAdd(&Df[row*N+col],ap[e]*4.0f);}
    }else{
        uint16_t*Db=(uint16_t*)D; float s=(col<N)?ws[col]:0.0f;
        for(int e=0;e<16;e++){int row=m_off+wmy*32+(e%4)+(e/4)*8+kg*4;if(row<M&&col<N)Db[row*N+col]=f32_to_bf16_bits(ap[e]*4.0f*s);}
    }
}



template<int BM, int BN, int NWFX, int NWFY, int BS, bool ATOMIC>
__global__ __launch_bounds__(BS) void gemm_db_r2(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    void* __restrict__ D, const float* __restrict__ ws,
    int M, int N, int K, int split_k) {
    __shared__ uint8_t As[N_BUF][BM][DB_BKP];
    __shared__ uint8_t Bs[N_BUF][BN][DB_BKP];
    int tid=threadIdx.x, wf=tid/WF_SIZE, lane=tid%WF_SIZE;
    int wmx=wf%NWFX, wmy=wf/NWFX, m_off=blockIdx.x*BM, n_off=blockIdx.y*BN;
    int mm=lane%32, kg=lane/32;
    __f16 acc0={},acc1={}; float* ap0=(float*)&acc0; float* ap1=(float*)&acc1;
    int kb_tot=(K+DB_BK-1)/DB_BK, kbps=(kb_tot+split_k-1)/split_k;
    int ks=blockIdx.z*kbps, ke=ks+kbps; if(ke>kb_tot) ke=kb_tot;
    int ni=ke-ks; if(ni<=0)return;
    const int ta=BM*DB_BK/16, tt=(BM+BN)*DB_BK/16, lpt=(tt+BS-1)/BS;
    {int kb_base=ks*DB_BK;
    for(int i=0;i<lpt;i++){int idx=i*BS+tid;
        if(idx<ta){int r=idx/(DB_BK/16),c16=idx%(DB_BK/16),c=c16*16;int gr=m_off+r,gc=kb_base+c;uint4 v={0,0,0,0};
        if(gr<M&&gc+15<K){v=*(const uint4*)&A[gr*K+gc];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
        else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gc+j;if(gr<M&&g<K){uint8_t b=A[gr*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&As[0][r][c]=v;}
        else if(idx<tt){int bi=idx-ta,r=bi/(DB_BK/16),c16=bi%(DB_BK/16),c=c16*16;int gn=n_off+r,gk=kb_base+c;uint4 v={0,0,0,0};
        if(gn<N&&gk+15<K){v=*(const uint4*)&B[gn*K+gk];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
        else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gk+j;if(gn<N&&g<K){uint8_t b=B[gn*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&Bs[0][r][c]=v;}
    }__syncthreads();}
    for(int it=0;it<ni-1;it++){
        int kb=ks+it, cb=it%N_BUF, nb=(it+1)%N_BUF, nb_base=(kb+1)*DB_BK;
        #pragma unroll
        for(int kc=0;kc<DB_BK/16;kc++){int ko=kc*16+kg*8;
            int64_t av=0;uint8_t*ab=(uint8_t*)&av;int ar=wmy*32+mm;
            *(uint32_t*)ab=*(uint32_t*)&As[cb][ar][ko];*(uint32_t*)(ab+4)=*(uint32_t*)&As[cb][ar][ko+4];
            int bc0=wmx*64+mm,bc1=wmx*64+32+mm;
            int64_t bv0=0;uint8_t*bb0=(uint8_t*)&bv0; *(uint32_t*)bb0=*(uint32_t*)&Bs[cb][bc0][ko];*(uint32_t*)(bb0+4)=*(uint32_t*)&Bs[cb][bc0][ko+4];
            int64_t bv1=0;uint8_t*bb1=(uint8_t*)&bv1; *(uint32_t*)bb1=*(uint32_t*)&Bs[cb][bc1][ko];*(uint32_t*)(bb1+4)=*(uint32_t*)&Bs[cb][bc1][ko+4];
            acc0=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv0,acc0,0,0,0);
            acc1=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv1,acc1,0,0,0);}
        for(int i=0;i<lpt;i++){int idx=i*BS+tid;
            if(idx<ta){int r=idx/(DB_BK/16),c16=idx%(DB_BK/16),c=c16*16;int gr=m_off+r,gc=nb_base+c;uint4 v={0,0,0,0};
            if(gr<M&&gc+15<K){v=*(const uint4*)&A[gr*K+gc];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
            else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gc+j;if(gr<M&&g<K){uint8_t b=A[gr*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&As[nb][r][c]=v;}
            else if(idx<tt){int bi=idx-ta,r=bi/(DB_BK/16),c16=bi%(DB_BK/16),c=c16*16;int gn=n_off+r,gk=nb_base+c;uint4 v={0,0,0,0};
            if(gn<N&&gk+15<K){v=*(const uint4*)&B[gn*K+gk];v.x=sanitize_fp8_u32(v.x);v.y=sanitize_fp8_u32(v.y);v.z=sanitize_fp8_u32(v.z);v.w=sanitize_fp8_u32(v.w);}
            else{uint8_t*p=(uint8_t*)&v;for(int j=0;j<16;j++){int g=gk+j;if(gn<N&&g<K){uint8_t b=B[gn*K+g];p[j]=(b==0x80u)?0u:b;}}} *(uint4*)&Bs[nb][r][c]=v;}
        }__syncthreads();}
    {int lb=(ni-1)%N_BUF;
    #pragma unroll
    for(int kc=0;kc<DB_BK/16;kc++){int ko=kc*16+kg*8;
        int64_t av=0;uint8_t*ab=(uint8_t*)&av;int ar=wmy*32+mm;
        *(uint32_t*)ab=*(uint32_t*)&As[lb][ar][ko];*(uint32_t*)(ab+4)=*(uint32_t*)&As[lb][ar][ko+4];
        int bc0=wmx*64+mm,bc1=wmx*64+32+mm;
        int64_t bv0=0;uint8_t*bb0=(uint8_t*)&bv0; *(uint32_t*)bb0=*(uint32_t*)&Bs[lb][bc0][ko];*(uint32_t*)(bb0+4)=*(uint32_t*)&Bs[lb][bc0][ko+4];
        int64_t bv1=0;uint8_t*bb1=(uint8_t*)&bv1; *(uint32_t*)bb1=*(uint32_t*)&Bs[lb][bc1][ko];*(uint32_t*)(bb1+4)=*(uint32_t*)&Bs[lb][bc1][ko+4];
        acc0=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv0,acc0,0,0,0);
        acc1=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av,bv1,acc1,0,0,0);}}
    int col0=n_off+wmx*64+mm,col1=n_off+wmx*64+32+mm;
    if(ATOMIC){float*Df=(float*)D;for(int e=0;e<16;e++){int row=m_off+wmy*32+(e%4)+(e/4)*8+kg*4;
        if(row<M&&col0<N)atomicAdd(&Df[row*N+col0],ap0[e]*4.0f);if(row<M&&col1<N)atomicAdd(&Df[row*N+col1],ap1[e]*4.0f);}}
    else{uint16_t*Db=(uint16_t*)D;float s0=(col0<N)?ws[col0]:0.0f,s1=(col1<N)?ws[col1]:0.0f;
    for(int e=0;e<16;e++){int row=m_off+wmy*32+(e%4)+(e/4)*8+kg*4;
        if(row<M&&col0<N)Db[row*N+col0]=f32_to_bf16_bits(ap0[e]*4.0f*s0);if(row<M&&col1<N)Db[row*N+col1]=f32_to_bf16_bits(ap1[e]*4.0f*s1);}}
}

extern "C" {
int mfma_fp8_gemm_r2(const void* a_ptr, const void* b_ptr, void* d_ptr,
    int M, int N, int K, int split_k, const void* weight_scale, void* stream) {
    hipStream_t hip_stream = (hipStream_t)stream;
    const float* ws = (const float*)weight_scale;
    dim3 grid((M+127)/128, (N+127)/128, split_k);
    if (split_k == 1)
        gemm_db_r2<128,128,2,4,512,false><<<grid,512,0,hip_stream>>>((const uint8_t*)a_ptr,(const uint8_t*)b_ptr,d_ptr,ws,M,N,K,split_k);
    else
        gemm_db_r2<128,128,2,4,512,true><<<grid,512,0,hip_stream>>>((const uint8_t*)a_ptr,(const uint8_t*)b_ptr,d_ptr,ws,M,N,K,split_k);
    return 0;
}
}
