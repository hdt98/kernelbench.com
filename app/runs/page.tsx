import { readdir } from "node:fs/promises"
import { join } from "node:path"
import Link from "next/link"
import { loadLeaderboard } from "@/app/_lib/data"
import { PageHead } from "@/app/_components/page-head"

// Solution source code lives in the GitHub repo; the site links each run
// keeps only the raw submitted-kernel files locally. Each row links to its HF
// trace, matching how /hard and /mega surface transcripts.
const HARD_TRACES_HF = "https://github.com/hdt98/kernelbench.com/tree/feat/amd-gpu-kernelbench/benchmarks/amd"

function traceUrl(runId: string): string {
  return `${HARD_TRACES_HF}/blob/main/${runId}.jsonl`
}

type RunRow = {
  run_id: string
  problem: string
  harness: string
  model: string
  effort: string
  correct: boolean
  has_solution: boolean
  failure_reason: string | null
  peak_fraction: number | null
  elapsed_seconds: number | null
}

type RunsIndexProps = {
  searchParams?: Promise<{
    harness?: string | string[]
  }>
}

async function loadAllRuns(): Promise<RunRow[]> {
  const lb = await loadLeaderboard()
  const solutionEntries = new Set<string>()
  try {
    const entries = await readdir(join(process.cwd(), "public/runs"))
    for (const n of entries) {
      if (n.endsWith("_solution.py.txt"))
        solutionEntries.add(n.slice(0, -"_solution.py.txt".length))
    }
  } catch {
    // no public/runs dir — page still works, just empty
  }

  const out: RunRow[] = []
  for (const m of lb.models) {
    for (const [problem, cell] of Object.entries(m.results)) {
      // AMD: show all runs from leaderboard, not just those with solution files in public/runs
      out.push({
        run_id: cell.run_id,
        problem,
        harness: m.harness,
        model: m.model,
        effort: m.effort,
        correct: cell.correct,
        has_solution: cell.has_solution,
        failure_reason: cell.failure_reason ?? null,
        peak_fraction: cell.peak_fraction,
        elapsed_seconds: cell.elapsed_seconds ?? null,
      })
    }
  }
  // Sort by scored peak_fraction desc, with correctness-only no-perf rows before FAIL/ERR.
  out.sort((a, b) => {
    const ap = a.correct ? a.peak_fraction ?? -0.5 : a.has_solution ? -1 : -2
    const bp = b.correct ? b.peak_fraction ?? -0.5 : b.has_solution ? -1 : -2
    return bp - ap
  })
  return out
}

function shortModel(harness: string, model: string, effort: string) {
  let m = model.replace("openrouter-pinned/", "or/")
  if (effort) m += ` [${effort}]`
  return m
}

function statusCell(r: RunRow) {
  if (r.correct) {
    return r.peak_fraction !== null ? (
      <span className="text-[var(--color-fg-bright)] tabular">
        {r.peak_fraction.toFixed(3)}
      </span>
    ) : (
      <span className="text-[var(--color-bad)]">
        {r.failure_reason === "benchmark_timeout" ? "BENCH" : "NO PERF"}
      </span>
    )
  }
  if (r.has_solution)
    return <span className="text-[var(--color-fg-dim)]">FAIL</span>
  return <span className="text-[var(--color-bad)]">ERR</span>
}

function harnessLabel(harness: string) {
  const labels: Record<string, string> = {
    claude: "Claude Code",
    codex: "Codex",
    opencode: "Opencode",
    droid: "droid",
    kimi: "kimi",
    cursor: "Cursor",
    gemini: "Gemini CLI",
    grok: "Grok Build",
    "zai-claude": "Claude Code",
    "minimax-claude": "Claude Code",
    "deepseek-claude": "Claude Code",
    "kimi-claude": "Claude Code",
  }
  if (harness.endsWith("-claude")) return "Claude Code"
  return labels[harness] ?? harness
}

function harnessClass(harness: string) {
  return harness === "opencode"
    ? "text-[var(--color-bad)]"
    : "text-[var(--color-fg-muted)]"
}

export default async function RunsIndex({ searchParams }: RunsIndexProps) {
  const allRuns = await loadAllRuns()
  const params = searchParams ? await searchParams : {}
  const rawHarness = Array.isArray(params.harness)
    ? params.harness[0]
    : params.harness
  const harnessFilter = rawHarness?.trim()
  const runs = harnessFilter
    ? allRuns.filter((r) => r.harness === harnessFilter)
    : allRuns

  const scored = runs.filter((r) => r.correct && r.peak_fraction !== null).length
  const noPerf = runs.filter((r) => r.correct && r.peak_fraction === null).length
  const fails = runs.filter((r) => !r.correct && r.has_solution).length
  const errs = runs.filter((r) => !r.correct && !r.has_solution).length
  const title = harnessFilter ? `Runs · ${harnessLabel(harnessFilter)}` : "Run index"

  return (
    <div className="space-y-6">
      <PageHead
        kicker="Index"
        title={title}
        sub={
          <>
            <strong>{runs.length}</strong> runs · {scored} scored · {noPerf} no perf ·{" "}
            {fails} fail · {errs} err
            {harnessFilter && (
              <>
                {" "}
                · filter{" "}
                <span className={harnessClass(harnessFilter)}>
                  harness={harnessLabel(harnessFilter)}
                </span>{" "}
                (<a href="/runs">clear</a>)
              </>
            )}
          </>
        }
        notes={
          <p>
            One row per (model, problem) cell, scored rows sorted by peak
            fraction desc. Click any row to view the run detail page with
            benchmark results and kernel approach. Source code lives in the{" "}
            <a href={HARD_TRACES_HF} target="_blank" rel="noreferrer">
              AMD benchmarks directory
            </a>{" "}
            on GitHub.
          </p>
        }
      />

      <section className="box overflow-x-auto">
        <table className="term tabular text-xs sm:text-sm">
          <thead>
            <tr>
              <th>peak</th>
              <th>problem</th>
              <th>model</th>
              <th>harness</th>
              <th>elapsed</th>
              <th>run id</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id}>
                <td className="text-right pr-4">{statusCell(r)}</td>
                <td className="text-[var(--color-fg)] whitespace-nowrap">
                  <Link
                    href={"/runs/mi325x/" + r.run_id}
                    className="no-underline hover:text-[var(--color-accent)]"
                  >
                    {r.problem}
                  </Link>
                </td>
                <td className="text-[var(--color-fg-bright)] whitespace-nowrap">
                  <Link
                    href={"/runs/mi325x/" + r.run_id}
                    className="no-underline hover:text-[var(--color-accent)]"
                  >
                    {shortModel(r.harness, r.model, r.effort)}
                  </Link>
                </td>
                <td className={harnessClass(r.harness)}>
                  {harnessLabel(r.harness)}
                </td>
                <td className="text-[var(--color-fg-muted)]">
                  {r.elapsed_seconds !== null
                    ? `${Math.round(r.elapsed_seconds / 60)}m`
                    : "-"}
                </td>
                <td className="text-[var(--color-fg-muted)] text-[10px]">
                  <Link
                    href={"/runs/mi325x/" + r.run_id}
                    className="no-underline hover:text-[var(--color-accent)] text-[10px]"
                  >
                    {r.run_id}
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
