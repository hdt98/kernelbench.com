import { notFound } from "next/navigation"
import { loadAmdDashboard } from "@/app/_lib/amd"

const AMD_GPU = "mi325x"

function fmtPct(value: number | null | undefined): string {
  return value == null ? "—" : (value * 100).toFixed(2) + "%"
}

export async function generateStaticParams() {
  const dashboard = await loadAmdDashboard()
  return dashboard.rows.map((row) => ({ name: row.slug }))
}

export default async function ProblemPage({
  params,
}: {
  params: Promise<{ name: string }>
}) {
  const { name } = await params
  const dashboard = await loadAmdDashboard()
  const row = dashboard.rows.find((entry) => entry.slug === name)
  if (!row) notFound()

  const meta = row.meta
  const status = row.status === "PASS" ? "PASS" : row.status === "PENDING" ? "PENDING" : "NOT ATTEMPTED"
  const statusClass = row.status === "PASS" ? "pass" : row.status === "PENDING" ? "pending" : "none"
  const bestRunHref = row.bestRunId ? "/runs/" + AMD_GPU + "/" + row.bestRunId : null

  return (
    <div className="space-y-6">
      <div className="problem-detail-header">
        <div className="problem-detail-breadcrumb">
          <a href="/" className="breadcrumb-link">Nexus KernelBench</a>
          <span className="breadcrumb-sep">/</span>
          <span className="breadcrumb-current">{row.displayName}</span>
        </div>
        <h1 className="problem-detail-title">{row.displayName}</h1>
        <div className="problem-detail-tags">
          <span className="tag tag-precision">{row.precision ?? "—"}</span>
          <span className="tag tag-regime">{row.regime ?? "—"}</span>
          <span className={"tag tag-status tag-status-" + statusClass}>{status}</span>
        </div>
      </div>

      <section className="problem-detail-section">
        <h2 className="section-title">Description</h2>
        <p className="problem-detail-desc">{meta.description}</p>
      </section>

      <section className="problem-detail-section">
        <h2 className="section-title">Solution Approach</h2>
        <p className="problem-detail-approach">{meta.approach}</p>
      </section>

      <section className="problem-detail-section">
        <h2 className="section-title">SOTA Ceiling</h2>
        <div className="space-y-2">
          <p className="problem-detail-sota">{meta.sota.name ?? "—"}</p>
          {meta.sota.url && (
            <a className="underlined-link" href={meta.sota.url} target="_blank" rel="noreferrer">
              SOTA reference ↗
            </a>
          )}
          {meta.sota.function && <p className="problem-detail-note">Function: {meta.sota.function}</p>}
          {meta.sota.notes && <p className="problem-detail-note">{meta.sota.notes}</p>}
          {meta.sota.deps.length > 0 && <p className="problem-detail-note">Deps: {meta.sota.deps.join(", ")}</p>}
          {meta.sota.referenceThroughputTflopsMi325x != null && (
            <p className="problem-detail-note">
              Reference TFLOPS: {meta.sota.referenceThroughputTflopsMi325x.toFixed(3)}
            </p>
          )}
          {meta.sota.referenceThroughputGbpsMi325x != null && (
            <p className="problem-detail-note">
              Reference GB/s: {meta.sota.referenceThroughputGbpsMi325x.toFixed(3)}
            </p>
          )}
          {meta.sota.referenceThroughputSpsMi325x != null && (
            <p className="problem-detail-note">
              Reference SPS: {meta.sota.referenceThroughputSpsMi325x.toFixed(3)}
            </p>
          )}
        </div>
      </section>

      <section className="problem-detail-section">
        <h2 className="section-title">Benchmark Config</h2>
        <div className="problem-detail-stats">
          <div className="detail-stat-card">
            <span className="detail-stat-label">Precision</span>
            <span className="detail-stat-value">{row.precision ?? "—"}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Regime</span>
            <span className="detail-stat-value">{row.regime ?? "—"}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Correct Trials</span>
            <span className="detail-stat-value">{meta.numCorrectTrials ?? "—"}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Perf Trials</span>
            <span className="detail-stat-value">{meta.numPerfTrials ?? "—"}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Language</span>
            <span className="detail-stat-value">{meta.language ?? "—"}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Triton Allowed</span>
            <span className="detail-stat-value">{meta.allowTriton == null ? "—" : meta.allowTriton ? "yes" : "no"}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Custom Kernel</span>
            <span className="detail-stat-value">{meta.requireCudaEvidence == null ? "—" : meta.requireCudaEvidence ? "required" : "not required"}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Best Peak</span>
            <span className="detail-stat-value">{fmtPct(row.bestPeakFraction)}</span>
          </div>
        </div>
      </section>

      <section className="problem-detail-section">
        <h2 className="section-title">Constraints</h2>
        <div className="space-y-2">
          <p className="problem-detail-note">Hardware: {meta.hardware.length ? meta.hardware.join(", ") : "—"}</p>
          <p className="problem-detail-note">Peak TFLOPS key: {meta.peakTflopsKey ?? "—"}</p>
          <p className="problem-detail-note">Peak bandwidth key: {meta.peakBandwidthKey ?? "—"}</p>
          {meta.toleranceLines.length > 0 && (
            <pre className="problem-detail-note whitespace-pre-wrap">{meta.toleranceLines.join("\n")}</pre>
          )}
          {meta.forbidden.length > 0 && (
            <p className="problem-detail-note">Forbidden: {meta.forbidden.join(", ")}</p>
          )}
        </div>
      </section>

      <section className="problem-detail-section">
        <h2 className="section-title">Performance</h2>
        <div className="problem-detail-stats">
          <div className="detail-stat-card">
            <span className="detail-stat-label">Status</span>
            <span className={"detail-stat-value detail-stat-" + statusClass}>{status}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Best Model</span>
            <span className="detail-stat-value">{row.bestModelLabel ?? "—"}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Solution Type</span>
            <span className="detail-stat-value">{row.solutionType}</span>
          </div>
          <div className="detail-stat-card">
            <span className="detail-stat-label">Best Run</span>
            <span className="detail-stat-value">
              {bestRunHref ? <a href={bestRunHref}>run ↗</a> : "—"}
            </span>
          </div>
        </div>
      </section>

      <section className="problem-detail-section">
        <h2 className="section-title">Hardware</h2>
        <div className="problem-detail-hw">
          <span>{dashboard.leaderboard.hardware.name}</span>
          <span>{dashboard.leaderboard.hardware.sm}</span>
          <span>{dashboard.leaderboard.hardware.vram_gb} GB VRAM</span>
          <span>{dashboard.leaderboard.hardware.peak_bandwidth_gb_s} GB/s peak bandwidth</span>
        </div>
      </section>
    </div>
  )
}
