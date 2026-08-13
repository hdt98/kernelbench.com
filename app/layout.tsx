import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { JetBrains_Mono } from "next/font/google"
import { SiteBrand } from "@/app/_components/site-brand"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
})

const mono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono-loaded",
})

export const metadata: Metadata = {
  metadataBase: new URL("https://kernelbench.com"),
  title: "kernelbench.com: AMD GPU Kernel Benchmark Results",
  description:
    "AMD Instinct GPU kernel benchmark results on MI325X (gfx942). Genuine Triton and HIP kernels.",
  authors: [{ name: "Elliot Arledge", url: "https://elliotarledge.com" }],
  creator: "Elliot Arledge",
  publisher: "kernelbench.com",
  keywords: [
    "GPU kernels",
    "ROCm",
    "HIP",
    "Triton",
    "benchmark",
    "coding agents",
    "LLM evaluation",
  ],
  openGraph: {
    title: "kernelbench.com: AMD GPU Kernel Benchmark Results",
    description:
      "AMD Instinct GPU kernel benchmark results on MI325X (gfx942). Genuine Triton and HIP kernels.",
    url: "https://kernelbench.com",
    siteName: "kernelbench.com",
  },
  other: {
    citation_title:
      "kernelbench.com: AMD GPU Kernel Benchmark Results",
    citation_author: "Elliot Arledge",
    citation_publication_date: "2026",
    citation_online_date: "2026",
    citation_fulltext_html_url: "https://kernelbench.com",
    citation_keywords:
      "GPU kernels; ROCm; HIP; Triton; coding agents; LLM evaluation; benchmark",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    // suppressHydrationWarning: browser extensions (Dark Reader, etc.) inject
    // attributes on <html>/<body> before React hydrates; without this the
    // Mac preview shows a noisy hydration overlay that is not app state.
    <html lang="en" className={inter.variable + " " + mono.variable} suppressHydrationWarning>
      <body className="min-h-screen bg-white text-zinc-950 antialiased" suppressHydrationWarning>
        <SiteBrand />
        <main className="mx-auto max-w-7xl px-6 lg:px-8 pt-8 pb-16">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}

function Footer() {
  return (
    <footer className="site-footer">
      <div className="site-footer-inner">
        <span className="site-footer-line">
          built by <a href="https://elliotarledge.com">elliot arledge</a>
          {" · "}
          <a href="https://github.com/hdt98/kernelbench.com">source</a>
        </span>
        <span className="site-footer-dim">
          independent site — not affiliated with Stanford KernelBench
        </span>
      </div>
    </footer>
  )
}
