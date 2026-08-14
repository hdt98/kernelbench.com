import { loadAllAmdDashboards, AMD_GPUS } from "@/app/_lib/amd"
import { AmdGpuDashboard } from "@/app/_components/amd-gpu-dashboard"

const citationGraph = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": "https://nexuskernel.com/#website",
      name: "nexuskernel.com",
      url: "https://nexuskernel.com",
      author: { "@type": "Organization", name: "One Mount Group", url: "https://onenexus-do.cloud" },
      description: "Open GPU kernel benchmark results for AMD Instinct GPUs.",
    },
  ],
}

export default async function HomePage() {
  const dashboards = await loadAllAmdDashboards()
  const gpus = AMD_GPUS.map((g) => ({ key: g.key, label: g.label }))

  return (
    <div className="space-y-8">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }} />
      <AmdGpuDashboard dashboards={dashboards} gpus={gpus} />
    </div>
  )
}
