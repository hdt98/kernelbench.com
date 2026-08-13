import { PROBLEM_LABELS } from "@/app/_lib/models"
import { loadLeaderboard } from "@/app/_lib/data"

const citationGraph = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": "https://kernelbench.com/#website",
      name: "kernelbench.com",
      url: "https://kernelbench.com",
      author: {
        "@type": "Person",
        name: "Elliot Arledge",
        url: "https://elliotarledge.com",
      },
      description:
        "Open agentic GPU kernel benchmark results, run transcripts, source repositories, and datasets.",
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

  return (
    <div className="space-y-8">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }}
      />

      {/* Hero section */}
      <section className="hero-section">
        <h1 className="hero-title">AMD KernelBench</h1>
        <p className="hero-subtitle">
          Agentic GPU kernel benchmark on AMD Instinct MI325X (gfx942, CDNA 3).
          Genuine Triton and HIP kernels — no reward hacking.
        </p>
        <div className="hero-stats">
          <div className="hero-stat">
            <span className="hero-stat-val">{hw.name}</span>
            <span className="hero-stat-label">Hardware</span>
          </div>
          <div className="hero-stat">
            <span className="hero-stat-val">{hw.sm}</span>
            <span className="hero-stat-label">Architecture</span>
          </div>
          <div className="hero-stat">
            <span className="hero-stat-val">{hw.vram_gb} GB</span>
            <span className="hero-stat-label">VRAM</span>
          </div>
          <div className="hero-stat">
            <span className="hero-stat-val">{hw.peak_bandwidth_gb_s} GB/s</span>
            <span className="hero-stat-label">Peak Bandwidth</span>
          </div>
        </div>
      </section>

      {/* Summary stats */}
      <section className="summary-section">
        <div className="summary-grid">
          <div className="summary-card">
            <span className="summary-num">{leaderboard.problems.length}</span>
            <span className="summary-label">Total Problems</span>
          </div>
          <div className="summary-card summary-card-pass">
            <span className="summary-num">{passingProblems.length}</span>
            <span className="summary-label">Passing</span>
          </div>
          <div className="summary-card summary-card-pending">
            <span className="summary-num">{pendingProblems.length}</span>
            <span className="summary-label">Pending</span>
          </div>
          <div className="summary-card summary-card-todo">
            <span className="summary-num">{unattempted.length}</span>
            <span className="summary-label">Not Attempted</span>
          </div>
        </div>
      </section>

      {/* Leaderboard table */}
      <section className="leaderboard-section">
        <h2 className="section-title">Leaderboard</h2>
        <div className="leaderboard-table-wrap">
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Problem</th>
                <th>Status</th>
                <th className="th-num">Peak Fraction</th>
                <th>Solution Type</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.problems.map((prob) => {
                const pp = leaderboard.per_problem[prob]
                const cell = model?.results?.[prob]
                const isPass = pp?.n_passed > 0
                const isPending = pp?.n_attempted > 0 && pp?.n_passed === 0
                const status = isPass ? "PASS" : isPending ? "PENDING" : "—"
                const frac = pp?.best_peak_fraction
                const fracStr = frac != null ? `${(frac * 100).toFixed(2)}%` : "—"
                const solType = cell?.has_solution
                  ? cell?.correct === true ? "Triton / HIP kernel" : "Written (unverified)"
                  : "Not attempted"
                return (
                  <tr key={prob} className={isPass ? "row-pass" : isPending ? "row-pending" : "row-none"}>
                    <td className="prob-name">{PROBLEM_LABELS[prob] ?? prob}</td>
                    <td className="prob-status">
                      <span className={`status-badge status-${isPass ? "pass" : isPending ? "pending" : "none"}`}>
                        {status}
                      </span>
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

    </div>
  )
}

