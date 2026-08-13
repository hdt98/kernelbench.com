import { readdir, readFile } from "node:fs/promises"
import { join } from "node:path"
import { loadLeaderboard, type Leaderboard } from "./data"
import { problemLabel } from "./models"

const AMD_ROOT = join(process.cwd(), "benchmarks/amd")
const AMD_PROBLEMS_ROOT = join(AMD_ROOT, "problems")
const AMD_LEADERBOARD = "benchmarks/amd/results/leaderboard.json"

export interface AmdSotaMeta {
  name: string | null
  url: string | null
  function: string | null
  deps: string[]
  notes: string | null
  referenceThroughputTflopsMi325x: number | null
  referenceThroughputGbpsMi325x: number | null
  referenceThroughputSpsMi325x: number | null
}

export interface AmdProblemMeta {
  slug: string
  name: string | null
  displayName: string
  precision: string | null
  regime: string | null
  flopsFormula: string | null
  bytesFormula: string | null
  hardware: string[]
  peakTflopsKey: string | null
  peakBandwidthKey: string | null
  toleranceLines: string[]
  forbidden: string[]
  allowTriton: boolean | null
  requireCudaEvidence: boolean | null
  language: string | null
  numCorrectTrials: number | null
  numPerfTrials: number | null
  sota: AmdSotaMeta
}

export interface AmdProblemRow {
  slug: string
  label: string
  displayName: string
  precision: string | null
  regime: string | null
  status: "PASS" | "PENDING" | "—"
  attempts: number
  passed: number
  bestPeakFraction: number | null
  bestModel: string | null
  bestModelLabel: string | null
  solutionType: string
  meta: AmdProblemMeta
}

export interface AmdDashboard {
  leaderboard: Leaderboard
  metas: Map<string, AmdProblemMeta>
  rows: AmdProblemRow[]
}

function unquote(value: string): string {
  const s = value.trim()
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
    return s.slice(1, -1)
  }
  return s
}

function stripComment(value: string): string {
  return value.replace(/\s+#.*$/, "").trim()
}

function scalar(text: string, key: string): string | null {
  const m = text.match(new RegExp(`^${key}:\\s*(.+)$`, "m"))
  if (!m) return null
  return unquote(stripComment(m[1]))
}

function numberScalar(text: string, key: string): number | null {
  const raw = scalar(text, key)
  if (raw == null || raw === "") return null
  const n = Number(raw)
  return Number.isFinite(n) ? n : null
}

function boolScalar(text: string, key: string): boolean | null {
  const raw = scalar(text, key)
  if (raw == null) return null
  if (raw === "true") return true
  if (raw === "false") return false
  return null
}

function inlineList(text: string, key: string): string[] {
  const m = text.match(new RegExp(`^${key}:\\s*\\[(.*)\\]\\s*$`, "m"))
  if (!m) return []
  const body = m[1].trim()
  if (!body) return []
  return body
    .split(",")
    .map((part) => unquote(part.trim()))
    .filter(Boolean)
}

function section(text: string, key: string): string | null {
  const lines = text.split(/\r?\n/)
  const start = lines.findIndex((line) => new RegExp(`^${key}:\\s*$`).test(line))
  if (start < 0) return null
  const out: string[] = []
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i]
    if (!line.trim()) {
      out.push("")
      continue
    }
    if (!line.startsWith("  ")) break
    out.push(line.slice(2))
  }
  return out.join("\n")
}

function blockList(text: string, key: string): string[] {
  const lines = text.split(/\r?\n/)
  const start = lines.findIndex((line) => new RegExp(`^${key}:\\s*$`).test(line))
  if (start < 0) return []
  const out: string[] = []
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i]
    if (!line.trim()) continue
    if (!line.startsWith("  - ")) break
    out.push(unquote(stripComment(line.slice(4))))
  }
  return out
}

function blockText(text: string, key: string): string | null {
  const lines = text.split(/\r?\n/)
  const start = lines.findIndex((line) => new RegExp(`^${key}:\\s*[>|]?\\s*$`).test(line))
  if (start < 0) return null
  const out: string[] = []
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i]
    if (!line.trim()) {
      out.push("")
      continue
    }
    if (!line.startsWith("  ")) break
    out.push(line.slice(2))
  }
  const textOut = out.join("\n").trimEnd()
  return textOut.length ? textOut : null
}

function parseSota(sectionText: string | null): AmdSotaMeta {
  if (!sectionText) {
    return {
      name: null,
      url: null,
      function: null,
      deps: [],
      notes: null,
      referenceThroughputTflopsMi325x: null,
      referenceThroughputGbpsMi325x: null,
      referenceThroughputSpsMi325x: null,
    }
  }
  return {
    name: scalar(sectionText, "name"),
    url: scalar(sectionText, "url"),
    function: scalar(sectionText, "function"),
    deps: blockList(sectionText, "deps"),
    notes: blockText(sectionText, "notes"),
    referenceThroughputTflopsMi325x: numberScalar(sectionText, "reference_throughput_tflops_mi325x"),
    referenceThroughputGbpsMi325x: numberScalar(sectionText, "reference_throughput_gbps_mi325x"),
    referenceThroughputSpsMi325x: numberScalar(sectionText, "reference_throughput_sps_mi325x"),
  }
}

export function parseAmdProblemYaml(slug: string, yamlText: string): AmdProblemMeta {
  return {
    slug,
    name: scalar(yamlText, "name"),
    displayName: scalar(yamlText, "display_name") ?? problemLabel(slug),
    precision: scalar(yamlText, "precision"),
    regime: scalar(yamlText, "regime"),
    flopsFormula: scalar(yamlText, "flops_formula"),
    bytesFormula: scalar(yamlText, "bytes_formula"),
    hardware: inlineList(yamlText, "hardware"),
    peakTflopsKey: scalar(yamlText, "peak_tflops_key"),
    peakBandwidthKey: scalar(yamlText, "peak_bandwidth_key"),
    toleranceLines: section(yamlText, "tolerance")?.split(/\r?\n/).filter(Boolean) ?? [],
    forbidden: blockList(yamlText, "forbidden"),
    allowTriton: boolScalar(yamlText, "allow_triton"),
    requireCudaEvidence: boolScalar(yamlText, "require_cuda_evidence"),
    language: scalar(yamlText, "language"),
    numCorrectTrials: numberScalar(yamlText, "num_correct_trials"),
    numPerfTrials: numberScalar(yamlText, "num_perf_trials"),
    sota: parseSota(section(yamlText, "sota")),
  }
}

async function loadProblemMeta(slug: string): Promise<AmdProblemMeta> {
  const path = join(AMD_PROBLEMS_ROOT, slug, "problem.yaml")
  try {
    return parseAmdProblemYaml(slug, await readFile(path, "utf-8"))
  } catch {
    return parseAmdProblemYaml(slug, "")
  }
}

export async function loadAmdLeaderboard(): Promise<Leaderboard> {
  return loadLeaderboard(AMD_LEADERBOARD)
}

export async function loadAmdProblemMetas(): Promise<Map<string, AmdProblemMeta>> {
  const metas = new Map<string, AmdProblemMeta>()
  const entries = await readdir(AMD_PROBLEMS_ROOT, { withFileTypes: true })
  await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => {
        metas.set(entry.name, await loadProblemMeta(entry.name))
      }),
  )
  return metas
}

function bestModelLabel(leaderboard: Leaderboard, modelId: string | null): string | null {
  if (!modelId) return null
  const found = leaderboard.models.find((m) => m.model === modelId)
  return found?.label ?? modelId
}

export async function loadAmdDashboard(): Promise<AmdDashboard> {
  const [leaderboard, metas] = await Promise.all([loadAmdLeaderboard(), loadAmdProblemMetas()])
  const rows = leaderboard.problems.map((slug) => {
    const meta = metas.get(slug) ?? parseAmdProblemYaml(slug, "")
    const stats = leaderboard.per_problem[slug] ?? {
      n_attempted: 0,
      n_passed: 0,
      best_peak_fraction: null,
      best_model: null,
      ranked_passes: [],
    }
    const bestModel = bestModelLabel(leaderboard, stats.best_model)
    const bestCell = stats.best_model
      ? leaderboard.models.find((m) => m.model === stats.best_model)?.results?.[slug] ?? null
      : null
    const solutionType = bestCell
      ? bestCell.has_solution
        ? bestCell.correct === true
          ? "Triton / HIP kernel"
          : "Written (unverified)"
        : "Not attempted"
      : stats.n_attempted > 0
        ? "Written (unverified)"
        : "Not attempted"
    return {
      slug,
      label: problemLabel(slug),
      displayName: meta.displayName,
      precision: meta.precision,
      regime: meta.regime,
      status: stats.n_passed > 0 ? "PASS" : stats.n_attempted > 0 ? "PENDING" : "—",
      attempts: stats.n_attempted,
      passed: stats.n_passed,
      bestPeakFraction: stats.best_peak_fraction,
      bestModel: stats.best_model,
      bestModelLabel: bestModel,
      solutionType,
      meta,
    } satisfies AmdProblemRow
  })
  return { leaderboard, metas, rows }
}
