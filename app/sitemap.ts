import type { MetadataRoute } from "next"
import { loadAmdDashboard } from "@/app/_lib/amd"

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const baseUrl = "https://kernelbench.com"
  const lastModified = new Date("2026-08-13")
  const dashboard = await loadAmdDashboard()

  const routes = [
    { path: "", priority: 1.0 },
    { path: "/amd", priority: 0.95 },
    { path: "/models", priority: 0.9 },
    { path: "/code", priority: 0.7 },
    ...dashboard.rows.map((row) => ({ path: `/problems/${row.slug}`, priority: 0.8 })),
  ]

  return routes.map(({ path, priority }) => ({
    url: `${baseUrl}${path}`,
    lastModified,
    changeFrequency: "weekly" as const,
    priority,
  }))
}
