import { readdir, readFile } from "node:fs/promises"
import { join } from "node:path"
import { loadLeaderboard, type Leaderboard } from "./data"
import { problemLabel } from "./models"

const AMD_ROOT = join(process.cwd(), "benchmarks/amd")
const AMD_PROBLEMS_ROOT = join(AMD_ROOT, "problems")
const AMD_LEADERBOARD = "benchmarks/amd/results/leaderboard.json"

const AMD_PROBLEM_COPY: Record<string, { description: string; approach: string }> = {
  "01_fp8_gemm": {
    description: "FP8 matrix multiplication using OCP e4m3 format, targeting MFMA units on gfx942.",
    approach: "Triton FP8 + MFMA hybrid",
  },
  "01_dequant_gemv": {
    description: "Gated W4A16 dequant-GEMV with group size 96. Fuses int4 unpack, group dequant, and gated GEMV.",
    approach: "Triton int4 dequant + GEMV",
  },
  "01_glm52_fused_moe": {
    description: "GLM-5.2 fused MoE layer with 256 routed experts, top-8 routing, 1 shared expert, SwiGLU activation.",
    approach: "PyTorch MoE (mem-efficient)",
  },
  "02_kda_cutlass": {
    description: "Kimi Delta Attention (chunk forward). Linear attention with delta decay.",
    approach: "Triton A-matrix kernel",
  },
  "02_deepseek_nsa": {
    description: "DeepSeek-style Native Sparse Attention with block selection and sliding window.",
    approach: "Vectorized PyTorch NSA",
  },
  "02_segmented_decay_scan": {
    description: "Segmented exponential-decay scan with per-token episode resets. Associative recurrence.",
    approach: "Triton sequential scan",
  },
  "03_megaqwen_decode": {
    description: "MegaQwen-style Qwen3-0.6B block decode. Multi-layer transformer with KV cache.",
    approach: "Eager PyTorch",
  },
  "03_paged_attention": {
    description: "Paged attention decode with block-table KV cache layout. Single-query attention.",
    approach: "Triton paged attention",
  },
  "03_topp_mask": {
    description: "Sort-free top-p (nucleus) mask. Binary-search threshold without sorting.",
    approach: "Triton binary-search",
  },
  "04_flash_attention": {
    description: "Causal FlashAttention forward pass with online softmax tiling.",
    approach: "Triton flash attention",
  },
  "04_grid_mingru_sps": {
    description: "Grid foraging RL environment + 3-layer MinGRU policy. Steps per second metric.",
    approach: "Triton GRU gate kernel",
  },
  "05_topk_bitonic": {
    description: "TopK selection via bitonic sort network. Parallel sorting on GPU.",
    approach: "Triton bitonic sort",
  },
  "06_sonic_moe_swiglu": {
    description: "Sonic-MoE up-projection: grouped GEMM + fused SwiGLU activation.",
    approach: "torch.mm + F.silu",
  },
  "07_w4a16_gemm": {
    description: "W4A16 weight-only quantized GEMM. INT4 weight dequant + BF16 GEMV.",
    approach: "Triton int4 dequant + GEMV",
  },
}

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
  description: string
  approach: string
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
  bestRunId: string | null
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
  const copy = AMD_PROBLEM_COPY[slug]
  return {
    slug,
    name: scalar(yamlText, "name"),
    displayName: scalar(yamlText, "display_name") ?? problemLabel(slug),
    description: copy?.description ?? scalar(yamlText, "display_name") ?? problemLabel(slug),
    approach: copy?.approach ?? "—",
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

function summarizeProblem(leaderboard: Leaderboard, slug: string) {
  let attempts = 0
  let passed = 0
  let bestModelId: string | null = null
  let bestCell: (typeof leaderboard.models)[number]["results"][string] | null = null

  for (const model of leaderboard.models) {
    const cell = model.results?.[slug]
    if (!cell) continue
    attempts += 1
    if (!cell.correct) continue
    passed += 1

    if (bestCell == null) {
      bestCell = cell
      bestModelId = model.model
      continue
    }

    const bestPeak = bestCell.peak_fraction
    const currentPeak = cell.peak_fraction
    if (currentPeak != null && (bestPeak == null || currentPeak > bestPeak)) {
      bestCell = cell
      bestModelId = model.model
    }
  }

  return { attempts, passed, bestModelId, bestCell }
}

export async function loadAmdDashboard(): Promise<AmdDashboard> {
  const [leaderboard, metas] = await Promise.all([loadAmdLeaderboard(), loadAmdProblemMetas()])
  const rows = leaderboard.problems.map((slug) => {
    const meta = metas.get(slug) ?? parseAmdProblemYaml(slug, "")
    const summary = summarizeProblem(leaderboard, slug)
    const bestModel = bestModelLabel(leaderboard, summary.bestModelId)
    const solutionType = summary.bestCell
      ? summary.bestCell.has_solution
        ? summary.bestCell.correct === true
          ? "Triton / HIP kernel"
          : "Written (unverified)"
        : "Not attempted"
      : summary.attempts > 0
        ? "Written (unverified)"
        : "Not attempted"
    return {
      slug,
      label: problemLabel(slug),
      displayName: meta.displayName,
      precision: meta.precision,
      regime: meta.regime,
      status: summary.passed > 0 ? "PASS" : summary.attempts > 0 ? "PENDING" : "—",
      attempts: summary.attempts,
      passed: summary.passed,
      bestPeakFraction: summary.bestCell?.peak_fraction ?? null,
      bestModel: summary.bestModelId,
      bestModelLabel: bestModel,
      bestRunId: summary.bestCell?.run_id ?? null,
      solutionType,
      meta,
    } satisfies AmdProblemRow
  })
  return { leaderboard, metas, rows }
}
