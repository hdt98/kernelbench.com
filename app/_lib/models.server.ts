import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { ModelIndex } from "./models"

// Server-side loader for the model index. Kept separate from ./models so the
// shared types/transforms stay importable from client components.

/** Models pulled from the site entirely (every chart, /models, model pages).
 *  Filtered at load so `kb publish` regenerating models.json can't
 *  resurrect them. */
const REMOVED_MODEL_SLUGS = new Set([
  "gemini-3.1-pro-preview",
  "gemini-3.5-flash",
  "claude-sonnet-5",
  "composer-2.5-fast",
  "deepseek-v4-pro",
  "minimax-m2.7",
  // Superseded by gpt-5.6-sol
  "gpt-5.5",
  // Retired preview ids — canonical is `hy3` only
  "hy3-preview",
  "tencent-hy3-preview",
  "tencent/hy3-preview",
])

// No module-level cache: Next dev (and prod workers) keep one module graph per
// route segment, so a `cached ??=` here pins each page to whatever models.json
// said at that segment's first request — the roster visibly desyncs across
// pages after a publish. The file is ~1 MB; reading it per request is noise.
export function loadModelIndex(): Promise<ModelIndex> {
  const mainP = readFile(join(process.cwd(), "public/data/models.json"), "utf8")
  const mi350xP = readFile(join(process.cwd(), "public/data/models_mi350x.json"), "utf8").catch(() => null)
  return Promise.all([mainP, mi350xP]).then(([raw, raw350]) => {
    const idx = JSON.parse(raw) as ModelIndex & { bench?: string }
    if (idx.bench === "amd") {
      const mi350 = raw350 ? (JSON.parse(raw350) as AmdLeaderboard) : null
      return synthesizeAmdIndex(idx as unknown as AmdLeaderboard, mi350)
    }
    idx.models = idx.models.filter((m) => !REMOVED_MODEL_SLUGS.has(m.slug))
    return idx
  })
}

type AmdLeaderboard = {
  bench: string
  hardware: {
    name: string
    sm: string
    vram_gb: number
    peak_bandwidth_gb_s: number
  }
  problems: string[]
  models: Array<{
    label: string
    harness: string
    model: string
    effort: string
    results: Record<
      string,
      {
        run_id: string
        correct: boolean | null
        has_solution: boolean
        peak_fraction: number | null
        failure_reason?: string | null
        retryable_infra_failure?: boolean | null
      }
    >
    pass_count: number
    total_runs: number
  }>
  per_problem: Record<
    string,
    {
      n_attempted: number
      n_passed: number
      best_peak_fraction: number | null
      best_model: string | null
      ranked_passes: { model: string; peak_fraction: number }[]
    }
  >
  methodology?: string
}

function synthesizeAmdIndex(board: AmdLeaderboard, mi350xBoard?: AmdLeaderboard | null): ModelIndex {
  const problems = board.problems
  const model = board.models[0]
  const cells: Record<string, any> = {}
  for (const prob of problems) {
    const c = model?.results?.[prob]
    cells[prob] = {
      run_id: c?.run_id ?? null,
      correct: Boolean(c?.correct),
      has_solution: Boolean(c?.has_solution),
      score: c?.peak_fraction ?? null,
      verdict: "unaudited",
      valid: Boolean(c?.correct && c?.peak_fraction != null),
      outcome: c?.correct === false ? "wrong" : c?.has_solution ? "other" : "empty",
      outcome_label:
        c?.failure_reason?.replace(/_/g, " ") ??
        (c?.correct ? "pass" : c?.has_solution ? "written" : "empty"),
      failure_reason: c?.failure_reason ?? null,
      elapsed_seconds: null,
      tok_s: null,
      ctx: undefined,
      framework: null,
      solution_url: null,
      trace_url: null,
      detail_url: null,
    }
  }

  const validScores = problems
    .map((prob) => {
      const c = model?.results?.[prob]
      const best = board.per_problem[prob]?.best_peak_fraction ?? null
      if (!c?.correct || c.peak_fraction == null || !best || best <= 0) return null
      return c.peak_fraction / best
    })
    .filter((v): v is number => v != null)
  const perf = validScores.length
    ? validScores.reduce((sum, value) => sum + value, 0) / validScores.length
    : null

  // Build MI350X GPU block if data is available
  let mi350xGpus: Record<string, any> = {}
  if (mi350xBoard) {
    const mi350xProblems = mi350xBoard.problems
    const mi350xModel = mi350xBoard.models[0]
    const mi350xCells: Record<string, any> = {}
    for (const prob of mi350xProblems) {
      const c = mi350xModel?.results?.[prob]
      mi350xCells[prob] = {
        run_id: c?.run_id ?? null,
        correct: Boolean(c?.correct),
        has_solution: Boolean(c?.has_solution),
        score: c?.peak_fraction ?? null,
        verdict: "unaudited",
        valid: Boolean(c?.correct && c?.peak_fraction != null),
        outcome: c?.correct === false ? "wrong" : c?.has_solution ? "other" : "empty",
        outcome_label:
          c?.failure_reason?.replace(/_/g, " ") ??
          (c?.correct ? "pass" : c?.has_solution ? "written" : "empty"),
        failure_reason: c?.failure_reason ?? null,
        elapsed_seconds: null,
        tok_s: null,
        ctx: undefined,
        framework: null,
        solution_url: null,
        trace_url: null,
        detail_url: null,
      }
    }
    mi350xGpus = {
      mi350x: {
        label: mi350xModel?.label ?? "AMD",
        harness: mi350xModel?.harness ?? "",
        effort: mi350xModel?.effort ?? "",
        passed: mi350xModel?.pass_count ?? 0,
        total_problems: mi350xProblems.length,
        perf: null,
        cells: mi350xCells,
      },
    }
  }

  return {
    generated: new Date().toISOString(),
    benches: {
      amd: {
        label: model?.label ?? "AMD",
        harness: model?.harness ?? "",
        effort: model?.effort ?? "",
        passed: model?.pass_count ?? 0,
        total_problems: problems.length,
        perf,
        cells,
        gpus: mi350xGpus,
      },
    },
    methodology:
      board.methodology ??
      "Nexus KernelBench: per-op kernel deck on AMD Instinct GPUs (MI325X & MI350X).",
    models: [
      {
        slug: model?.model ?? "amd",
        name: model?.label ?? "AMD",
        lab: "AMD",
        benches: {
          amd: {
            label: model?.label ?? "AMD",
            harness: model?.harness ?? "",
            effort: model?.effort ?? "",
            passed: model?.pass_count ?? 0,
            total_problems: problems.length,
            perf,
            cells,
            gpus: mi350xGpus,
          },
        },
        legacy: {},
        totals: { audited: 0, flagged: 0 },
      } as ModelIndex["models"][number],
    ],
  }
}
