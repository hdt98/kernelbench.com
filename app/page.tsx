import { PROBLEM_LABELS } from "@/app/_lib/models"
import { loadLeaderboard } from "@/app/_lib/data"

const PROBLEM_META: Record<string, { precision: string; regime: string; desc: string }> = {
  "01_fp8_gemm": { precision: "FP8 e4m3", regime: "compute", desc: "FP8 matrix multiplication using OCP e4m3 format, targeting MFMA units on gfx942." },
  "01_dequant_gemv": { precision: "INT4+BF16", regime: "memory", desc: "Gated W4A16 dequant-GEMV with group size 96. Fuses int4 unpack, group dequant, and gated GEMV." },
  "01_glm52_fused_moe": { precision: "BF16", regime: "compute", desc: "GLM-5.2 fused MoE layer with 256 routed experts, top-8 routing, 1 shared expert, SwiGLU activation." },
  "02_kda_cutlass": { precision: "BF16", regime: "compute", desc: "Kimi Delta Attention (chunk forward). Linear attention with delta decay." },
  "02_deepseek_nsa": { precision: "BF16", regime: "compute", desc: "DeepSeek-style Native Sparse Attention with block selection and sliding window." },
  "02_segmented_decay_scan": { precision: "BF16", regime: "memory", desc: "Segmented exponential-decay scan with per-token episode resets. Associative recurrence." },
  "03_megaqwen_decode": { precision: "BF16", regime: "throughput", desc: "MegaQwen-style Qwen3-0.6B block decode. Multi-layer transformer with KV cache." },
  "03_paged_attention": { precision: "BF16", regime: "memory", desc: "Paged attention decode with block-table KV cache layout. Single-query attention." },
  "03_topp_mask": { precision: "FP32", regime: "memory", desc: "Sort-free top-p (nucleus) mask. Binary-search threshold without sorting." },
  "04_flash_attention": { precision: "BF16", regime: "compute", desc: "Causal FlashAttention forward pass with online softmax tiling." },
  "04_grid_mingru_sps": { precision: "FP32", regime: "throughput", desc: "Grid foraging RL environment + 3-layer MinGRU policy. Steps per second metric." },
  "05_topk_bitonic": { precision: "FP32", regime: "memory", desc: "TopK selection via bitonic sort network. Parallel sorting on GPU." },
  "06_sonic_moe_swiglu": { precision: "BF16", regime: "compute", desc: "Sonic-MoE up-projection: grouped GEMM + fused SwiGLU activation." },
  "07_w4a16_gemm": { precision: "INT4+BF16", regime: "memory", desc: "W4A16 weight-only quantized GEMM. INT4 weight dequant + BF16 GEMV." },
}

const citationGraph = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": "https://kernelbench.com/#website",
      name: "kernelbench.com",
      url: "https://kernelbench.com",
      author: { "@type": "Person", name: "Elliot Arledge", url: "https://elliotarledge.com" },
      description: "Open agentic GPU kernel benchmark results for AMD Instinct GPUs.",
    },
  ],
}

export default async function HomePage() {
  const leaderboard = await loadLeaderboard()
  const hw = leaderboard.hardware
  const model = leaderboard.models[0]
  const passingProblems = leaderboard.problems.filter(
    (p) => leaderboard.per_problem[p]?.n_passed > 0
  )
  const pendingProblems = leaderboard.problems.filter(
    (p) => leaderboard.per_problem[p]?.n_attempted > 0 && leaderboard.per_problem[p]?.n_passed === 0
  )
  const unattempted = leaderboard.problems.filter(
    (p) => leaderboard.per_problem[p]?.n_attempted === 0
  )
  const geomeanPeaks = passingProblems
    .map((p) => leaderboard.per_problem[p]?.best_peak_fraction)
    .filter((v): v is number => v != null && v > 0)
  const geomean = geomeanPeaks.length > 0
    ? Math.exp(geomeanPeaks.reduce((s, v) => s + Math.log(v), 0) / geomeanPeaks.length)
    : 0

  return (
    <div className="space-y-8">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }} />

      <section className="hero-section">
        <h1 className="hero-title">AMD KernelBench</h1>
        <p className="hero-subtitle">
          Agentic GPU kernel benchmark on AMD Instinct MI325X (gfx942, CDNA 3).
          Genuine Triton and HIP kernels — no reward hacking.
        </p>
        <div className="hero-stats">
          <div className="hero-stat"><span className="hero-stat-val">{hw.name}</span><span className="hero-stat-label">Hardware</span></div>
          <div className="hero-stat"><span className="hero-stat-val">{hw.sm}</span><span className="hero-stat-label">Architecture</span></div>
          <div className="hero-stat"><span className="hero-stat-val">{hw.vram_gb} GB</span><span className="hero-stat-label">VRAM</span></div>
          <div className="hero-stat"><span className="hero-stat-val">{hw.peak_bandwidth_gb_s} GB/s</span><span className="hero-stat-label">Peak Bandwidth</span></div>
        </div>
      </section>

      <section className="summary-section">
        <div className="summary-grid">
          <div className="summary-card"><span className="summary-num">{leaderboard.problems.length}</span><span className="summary-label">Total Problems</span></div>
          <div className="summary-card summary-card-pass"><span className="summary-num">{passingProblems.length}</span><span className="summary-label">Passing</span></div>
          <div className="summary-card summary-card-pending"><span className="summary-num">{pendingProblems.length}</span><span className="summary-label">Pending</span></div>
          <div className="summary-card summary-card-todo"><span className="summary-num">{unattempted.length}</span><span className="summary-label">Not Attempted</span></div>
          {geomean > 0 && (
            <div className="summary-card summary-card-pass"><span className="summary-num">{(geomean * 100).toFixed(2)}%</span><span className="summary-label">Geomean Peak</span></div>
          )}
        </div>
      </section>

      <section className="chart-section">
        <h2 className="section-title">Peak Fraction by Problem</h2>
        <div className="bar-chart">
          {passingProblems
            .filter((p) => (leaderboard.per_problem[p]?.best_peak_fraction ?? 0) > 0)
            .map((prob) => {
              const frac = leaderboard.per_problem[prob]?.best_peak_fraction ?? 0
              const pct = (frac * 100).toFixed(2)
              const barWidth = Math.max(frac * 1000, 1)
              return (
                <div key={prob} className="bar-row">
                  <a href={"/problems/" + prob} className="bar-label">{PROBLEM_LABELS[prob] ?? prob}</a>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: barWidth + "%" }} />
                  </div>
                  <span className="bar-value">{pct}%</span>
                </div>
              )
            })}
        </div>
      </section>

      <section className="leaderboard-section">
        <h2 className="section-title">Problem Results</h2>
        <div className="leaderboard-table-wrap">
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Problem</th>
                <th>Precision</th>
                <th>Regime</th>
                <th>Status</th>
                <th className="th-num">Peak Fraction</th>
                <th>Solution</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.problems.map((prob) => {
                const pp = leaderboard.per_problem[prob]
                const cell = model?.results?.[prob]
                const meta = PROBLEM_META[prob]
                const isPass = pp?.n_passed > 0
                const isPending = pp?.n_attempted > 0 && pp?.n_passed === 0
                const status = isPass ? "PASS" : isPending ? "PENDING" : "—"
                const frac = pp?.best_peak_fraction
                const fracStr = frac != null ? `${(frac * 100).toFixed(2)}%` : "—"
                const solType = cell?.has_solution
                  ? cell?.correct === true ? "Triton / HIP" : "Written"
                  : "—"
                return (
                  <tr key={prob} className={isPass ? "row-pass" : isPending ? "row-pending" : "row-none"}>
                    <td className="prob-name"><a href={"/problems/" + prob}>{PROBLEM_LABELS[prob] ?? prob}</a></td>
                    <td className="prob-precision">{meta?.precision ?? "—"}</td>
                    <td className="prob-regime">{meta?.regime ?? "—"}</td>
                    <td className="prob-status">
                      <span className={`status-badge status-${isPass ? "pass" : isPending ? "pending" : "none"}`}>{status}</span>
                    </td>
                    <td className="td-num">{fracStr}</td>
                    <td className="sol-type">{solType}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="methodology-section">
        <h2 className="section-title">Methodology</h2>
        <div className="methodology-grid">
          <div className="methodology-card">
            <h3>Kernel Writing</h3>
            <p>Each problem provides a reference implementation (naive PyTorch) and a SOTA ceiling. The task is to write a custom GPU kernel that matches the reference output within tolerance and maximizes throughput against the hardware roofline.</p>
          </div>
          <div className="methodology-card">
            <h3>Correctness Gate</h3>
            <p>Solutions must pass check.py which validates output correctness across multiple shapes, seeds, and numeric stress cases. Floating-point outputs use explicit per-dtype tolerances. No reward hacking.</p>
          </div>
          <div className="methodology-card">
            <h3>Roofline Benchmark</h3>
            <p>Performance is measured by benchmark.py using median timing over 15 trials. The peak fraction is the ratio of achieved TFLOPS (or GBPS for memory-bound problems) to the hardware peak.</p>
          </div>
          <div className="methodology-card">
            <h3>AMD MI325X</h3>
            <p>The MI325X is based on gfx942 (CDNA 3) with 288 GB HBM3e and 8 TB/s peak bandwidth. Kernels use Triton on ROCm and PyTorch HIP backends.</p>
          </div>
        </div>
      </section>

      <section className="leaderboard-section">
        <h2 className="section-title">Problem Descriptions</h2>
        <div className="problem-descriptions">
          {leaderboard.problems.map((prob) => {
            const meta = PROBLEM_META[prob]
            const pp = leaderboard.per_problem[prob]
            const isPass = pp?.n_passed > 0
            if (!meta) return null
            return (
              <div key={prob} className={`problem-card ${isPass ? "problem-card-pass" : ""}`}>
                <div className="problem-card-header">
                  <a href={"/problems/" + prob} className="problem-card-name">{PROBLEM_LABELS[prob] ?? prob}</a>
                  <span className="problem-card-tags">
                    <span className="tag tag-precision">{meta.precision}</span>
                    <span className="tag tag-regime">{meta.regime}</span>
                  </span>
                </div>
                <p className="problem-card-desc">{meta.desc}</p>
              </div>
            )
          })}
        </div>
      </section>
    </div>
  )
}

