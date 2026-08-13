import { notFound } from "next/navigation"
import { loadLeaderboard, type Leaderboard } from "@/app/_lib/data"
import { PROBLEM_LABELS } from "@/app/_lib/models"

const PROBLEM_META: Record<string, { precision: string; regime: string; desc: string; approach: string }> = {
  "01_fp8_gemm": { precision: "FP8 e4m3", regime: "compute", desc: "FP8 matrix multiplication using OCP e4m3 format, targeting MFMA units on gfx942.", approach: "Triton FP8 GEMM with FNUZ scale correction and MFMA hybrid. Handles the FNUZ vs OCP exponent bias difference (bias=8 vs 7) by applying a 4x scale correction. Sanitizes OCP -0 (0x80) to 0x00 since it's NaN in FNUZ." },
  "01_dequant_gemv": { precision: "INT4+BF16", regime: "memory", desc: "Gated W4A16 dequant-GEMV with group size 96. Fuses int4 unpack, group dequant, and gated GEMV.", approach: "Triton kernel for int4 dequantization + GEMV with group-wise scales. Handles group size 96 alignment. Falls back to PyTorch for M<=4 cases." },
  "01_glm52_fused_moe": { precision: "BF16", regime: "compute", desc: "GLM-5.2 fused MoE layer with 256 routed experts, top-8 routing, 1 shared expert, SwiGLU activation.", approach: "PyTorch implementation matching reference semantics. Iterates over 256 experts, gathering tokens per expert, computing SwiGLU gate*up projection and down projection. Uses p.data.zero_() init to avoid nn.init.normal_ segfault." },
  "02_kda_cutlass": { precision: "BF16", regime: "compute", desc: "Kimi Delta Attention (chunk forward). Linear attention with delta decay.", approach: "Pure PyTorch einsum/softmax implementation. Matches reference semantics with chunk-based delta attention computation." },
  "02_deepseek_nsa": { precision: "BF16", regime: "compute", desc: "DeepSeek-style Native Sparse Attention with block selection and sliding window.", approach: "Vectorized PyTorch implementation of NSA. Computes block importance via mean Q*K scores, selects top-n blocks, unions with sliding window, then does sparse softmax attention. Chunked over query positions to manage memory." },
  "02_segmented_decay_scan": { precision: "BF16", regime: "memory", desc: "Segmented exponential-decay scan with per-token episode resets. Associative recurrence.", approach: "Triton kernel with runtime for-loop over T timesteps. Each program handles one (batch, d_block) pair, iterating sequentially through time with exponential decay accumulation." },
  "03_megaqwen_decode": { precision: "BF16", regime: "throughput", desc: "MegaQwen-style Qwen3-0.6B block decode. Multi-layer transformer with KV cache.", approach: "Eager PyTorch matching reference numerics. RMSNorm, QKV projection, RoPE, GQA attention, SwiGLU MLP. Uses p.data.zero_() init. Sequential prefill + decode with KV cache management." },
  "03_paged_attention": { precision: "BF16", regime: "memory", desc: "Paged attention decode with block-table KV cache layout. Single-query attention.", approach: "Pure PyTorch implementation with matmul/softmax. Handles paged KV cache layout with block tables." },
  "03_topp_mask": { precision: "FP32", regime: "memory", desc: "Sort-free top-p (nucleus) mask. Binary-search threshold without sorting.", approach: "Triton kernel using 60-iteration binary search on probability threshold. Updates lo/hi bounds using conditional masking (no break statements). Sort-free top-p via threshold convergence." },
  "04_flash_attention": { precision: "BF16", regime: "compute", desc: "Causal FlashAttention forward pass with online softmax tiling.", approach: "Triton flash attention with online softmax. BQ=32, BK=64, BD=128, num_stages=1 to fit within gfx942's 65536-byte shared memory limit. Causal masking via triangular mask." },
  "04_grid_mingru_sps": { precision: "FP32", regime: "throughput", desc: "Grid foraging RL environment + 3-layer MinGRU policy. Steps per second metric.", approach: "PyTorch policy_forward with F.linear for MinGRU layers. env_step uses PyTorch with exact LCG RNG matching. Triton kernel defined for evidence but policy uses PyTorch path." },
  "05_topk_bitonic": { precision: "FP32", regime: "memory", desc: "TopK selection via bitonic sort network. Parallel sorting on GPU.", approach: "Triton bitonic sort kernel. Parallel sorting network with O(log^2 n) stages. Each stage compares and swaps elements." },
  "06_sonic_moe_swiglu": { precision: "BF16", regime: "compute", desc: "Sonic-MoE up-projection: grouped GEMM + fused SwiGLU activation.", approach: "PyTorch torch.mm + F.silu implementation. Matches reference interface with expert offsets. Uses p.data.zero_() init." },
  "07_w4a16_gemm": { precision: "INT4+BF16", regime: "memory", desc: "W4A16 weight-only quantized GEMM. INT4 weight dequant + BF16 GEMV.", approach: "Triton kernel for INT4 weight dequantization + BF16 GEMV. Unpacks 4-bit weights, applies group scales, then matmul with BF16 activation." },
}

export async function generateStaticParams() {
  const lb = await loadLeaderboard()
  return lb.problems.map((name) => ({ name }))
}

export default async function ProblemPage({
  params,
}: {
  params: Promise<{ name: string }>
}) {
  const { name } = await params
  const lb = await loadLeaderboard()
  const meta = PROBLEM_META[name]
  if (!meta) notFound()

  const pp = lb.per_problem[name]
  const model = lb.models[0]
  const cell = model?.results?.[name]
  const isPass = pp?.n_passed > 0
  const isPending = pp?.n_attempted > 0 && pp?.n_passed === 0
  const frac = pp?.best_peak_fraction
  const fracStr = frac != null ? `(frac * 100).toFixed(2)}%` : "—"
  const status = isPass ? "PASS" : isPending ? "PENDING" : "NOT ATTEMPTED"

  return (
    <div className="space-y-6">
      <div className="problem-detail-header">
        <div className="problem-detail-breadcrumb">
          <a href="/" className="breadcrumb-link">AMD KernelBench</a>
          <span className="breadcrumb-sep">/</span>
          <span className="breadcrumb-current">{PROBLEM_LABELS[name] ?? name}</span>
        </div>
        <h1 className="problem-detail-title">{PROBLEM_LABELS[name] ?? name}</h1>
        <div className="problem-detail-tags">
          <span className="tag tag-precision">{meta.precision}</span>
          <span className="tag tag-regime">{meta.regime}</span>
          <span className={`tag tag-status tag-status-${isPass ? "pass" : isPending ? "pending" : "none"}`}>{status}</span>
        </div>
      </div>

      <section className="problem-detail-section">
        <h2 className="section-title">Description</h2>
        <p className="problem-detail-desc">{meta.desc}</p>
      </section>

      <section className="problem-detail-section">
        <h2 className="section-title">Solution Approach</h2>
        <p className="problem-detail-approach">{meta.approach}</p>
      </section>

      <section className="problem-detail-section">
        <h2 className="section-title">Performance</h2>
        <div className="problem-detail-stats">
          <div className="detail-stat-card">
            <span className="detail-stat-label">Status</span>
            <span className={`detail-stat-value detail-stat-${isPass ? "pass" : isPending ? "pending" : "none"}`}>{status}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Peak Fraction</span>
            <span className="detail-stat-value">{fracStr}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Solution Type</span>
            <span className="detail-stat-value">{cell?.has_solution ? "Custom Kernel" : "—"}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Correctness</span>
            <span className={`detail-stat-value detail-stat-${cell?.correct === true ? "pass" : cell?.correct === false ? "none" : "pending"}`}>{cell?.correct === true ? "Verified" : cell?.correct === false ? "Failed" : "Pending"}</span>
          </div>
        </div>
        {cell?.failure_reason && (
          <p className="problem-detail-note">Note: {cell.failure_reason}</p>
        )}
      </section>

      <section className="problem-detail-section">
        <h2 className="section-title">Hardware</h2>
        <div className="problem-detail-hw">
          <span>{lb.hardware.name}</span>
          <span>{lb.hardware.sm}</span>
          <span>{lb.hardware.vram_gb} GB VRAM</span>
          <span>{lb.hardware.peak_bandwidth_gb_s} GB/s peak bandwidth</span>
        </div>
      </section>
    </div>
  )
}

