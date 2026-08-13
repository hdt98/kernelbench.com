import { loadAmdDashboard } from "@/app/_lib/amd"
import { PROBLEM_LABELS } from "@/app/_lib/models"

const citationGraph = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": "https://nexuskernel.com/#website",
      name: "nexuskernel.com",
      url: "https://nexuskernel.com",
      author: { "@type": "Person", name: "Elliot Arledge", url: "https://elliotarledge.com" },
      description: "Open GPU kernel benchmark results for AMD Instinct GPUs.",
    },
  ],
}

function fmtPct(value: number | null | undefined): string {
  return value == null ? "—" : (value * 100).toFixed(2) + "%"
}

function modelGeomean(
  results: Record<string, { correct: boolean | null; peak_fraction: number | null }>,
  problems: string[],
): number {
  const peaks = problems
    .filter((p) => results[p]?.correct && results[p]?.peak_fraction != null && results[p]!.peak_fraction! > 0)
    .map((p) => results[p]!.peak_fraction!)
  if (peaks.length === 0) return 0
  return Math.exp(peaks.reduce((s, v) => s + Math.log(v), 0) / peaks.length)
}

export default async function HomePage() {
  const dashboard = await loadAmdDashboard()
  const leaderboard = dashboard.leaderboard
  const rowsBySlug = new Map(dashboard.rows.map((row) => [row.slug, row]))

  const passingRows = dashboard.rows.filter((row) => row.status === "PASS")
  const pendingRows = dashboard.rows.filter((row) => row.status === "PENDING")
  const unattemptedRows = dashboard.rows.filter((row) => row.status === "—")
  const geomeanPeaks = passingRows
    .map((row) => row.bestPeakFraction)
    .filter((value): value is number => value != null && value > 0)
  const geomean =
    geomeanPeaks.length > 0
      ? Math.exp(geomeanPeaks.reduce((sum, value) => sum + Math.log(value), 0) / geomeanPeaks.length)
      : 0

  const rankedModels = [...leaderboard.models].sort(
    (a, b) => modelGeomean(b.results, leaderboard.problems) - modelGeomean(a.results, leaderboard.problems),
  )

  return (
    <div className="space-y-8">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }} />

      <section className="hero-section" id="amd">
        <h1 className="hero-title">AMD KernelBench</h1>
        <p className="hero-subtitle">
          GPU kernel benchmark on AMD Instinct MI325X (gfx942, CDNA 3).
          Genuine Triton and HIP kernels — no reward hacking.
        </p>
        <div className="hero-stats">
          <div className="hero-stat"><span className="hero-stat-val">{leaderboard.hardware.name}</span><span className="hero-stat-label">Hardware</span></div>
          <div className="hero-stat"><span className="hero-stat-val">{leaderboard.hardware.sm}</span><span className="hero-stat-label">Architecture</span></div>
          <div className="hero-stat"><span className="hero-stat-val">{leaderboard.hardware.vram_gb} GB</span><span className="hero-stat-label">VRAM</span></div>
          <div className="hero-stat"><span className="hero-stat-val">{leaderboard.hardware.peak_bandwidth_gb_s} GB/s</span><span className="hero-stat-label">Peak Bandwidth</span></div>
        </div>
      </section>

      <section className="summary-section">
        <div className="summary-grid">
          <div className="summary-card"><span className="summary-num">{leaderboard.problems.length}</span><span className="summary-label">Total Problems</span></div>
          <div className="summary-card summary-card-pass"><span className="summary-num">{passingRows.length}</span><span className="summary-label">Passing</span></div>
          <div className="summary-card summary-card-pending"><span className="summary-num">{pendingRows.length}</span><span className="summary-label">Pending</span></div>
          <div className="summary-card summary-card-todo"><span className="summary-num">{unattemptedRows.length}</span><span className="summary-label">Not Attempted</span></div>
          {geomean > 0 && (
            <div className="summary-card summary-card-pass"><span className="summary-num">{(geomean * 100).toFixed(2)}%</span><span className="summary-label">Geomean Peak</span></div>
          )}
        </div>
      </section>

      <section className="chart-section">
        <h2 className="section-title">Peak Fraction by Problem</h2>
        <div className="bar-chart">
          {passingRows
            .filter((row) => (row.bestPeakFraction ?? 0) > 0)
            .map((row) => {
              const frac = row.bestPeakFraction ?? 0
              const barWidth = Math.max(frac * 1000, 1)
              return (
                <div key={row.slug} className="bar-row">
                  <a href={"/problems/" + row.slug} className="bar-label">{row.displayName}</a>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: barWidth + "%" }} />
                  </div>
                  <span className="bar-value">{fmtPct(frac)}</span>
                </div>
              )
            })}
        </div>
      </section>

      <section className="leaderboard-section">
        <h2 className="section-title">Model Leaderboard</h2>
        <div className="leaderboard-table-wrap">
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Harness</th>
                <th>Effort</th>
                <th className="th-num">Pass</th>
                <th className="th-num">Geomean Peak</th>
              </tr>
            </thead>
            <tbody>
              {rankedModels.map((m, i) => {
                const gm = modelGeomean(m.results, leaderboard.problems)
                return (
                  <tr key={m.model} className="row-pass">
                    <td className="rank-cell">{i + 1}</td>
                    <td className="model-name">{m.label}</td>
                    <td className="model-harness">{m.harness}</td>
                    <td className="model-harness">{m.effort || "—"}</td>
                    <td className="td-num">{m.pass_count}/{m.total_runs}</td>
                    <td className="td-num">{(gm * 100).toFixed(2)}%</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
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
                {rankedModels.map((m) => (
                  <th key={m.model} className="th-model">{m.label}</th>
                ))}
                <th>SOTA Ceiling</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.problems.map((prob) => {
                const row = rowsBySlug.get(prob)
                if (!row) return null
                const isPass = row.status === "PASS"
                const isPending = row.status === "PENDING"
                return (
                  <tr key={prob} className={isPass ? "row-pass" : isPending ? "row-pending" : "row-none"}>
                    <td className="prob-name"><a href={"/problems/" + prob}>{row.displayName}</a></td>
                    <td className="prob-precision">{row.precision ?? "—"}</td>
                    <td className="prob-regime">{row.regime ?? "—"}</td>
                    {rankedModels.map((m) => {
                      const cell = m.results[prob]
                      if (!cell || !cell.correct) {
                        return <td key={m.model} className="td-num cell-empty">—</td>
                      }
                      const frac = cell.peak_fraction
                      if (frac == null) {
                        return <td key={m.model} className="td-num cell-pass-noperf">PASS</td>
                      }
                      const pctStr = (frac * 100).toFixed(2) + "%"
                      const cellClass = frac >= 0.1 ? "cell-good" : frac >= 0.05 ? "cell-ok" : frac >= 0.01 ? "cell-low" : "cell-vlow"
                      return (
                        <td key={m.model} className={"td-num " + cellClass}>
                          <a href={"/runs/mi325x/" + cell.run_id} className="cell-link">{pctStr}</a>
                        </td>
                      )
                    })}
                    <td className="prob-sota">{row.meta.sota?.name ?? row.meta.sota?.function ?? "—"}</td>
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
            <p>Performance is measured by benchmark.py using median timing over multiple trials (problem-dependent). The peak fraction is the ratio of achieved TFLOPS (or GBPS for memory-bound problems) to the hardware peak.</p>
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
            const row = rowsBySlug.get(prob)
            if (!row) return null
            const isPass = row.status === "PASS"
            return (
              <div key={prob} className={"problem-card " + (isPass ? "problem-card-pass" : "")}>
                <div className="problem-card-header">
                  <a href={"/problems/" + prob} className="problem-card-name">{row.displayName}</a>
                  <span className="problem-card-tags">
                    <span className="tag tag-precision">{row.precision ?? "—"}</span>
                    <span className="tag tag-regime">{row.regime ?? "—"}</span>
                  </span>
                </div>
                <p className="problem-card-desc">{row.meta.description ?? row.meta.name ?? ""}</p>
              </div>
            )
          })}
        </div>
      </section>
    </div>
  )
}
