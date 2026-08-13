// kernelbench.com header: sticky, backdrop-blur, clean zinc palette.

export function SiteBrand() {
  return (
    <header className="sticky top-0 z-50 border-b border-zinc-100 bg-white/85 backdrop-blur-xl">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6 lg:px-8" aria-label="Primary navigation">
        <a className="flex items-center gap-2.5 text-[17px] font-semibold tracking-tight text-zinc-950 no-underline" href="/" aria-label="kernelbench.com home">
          <span className="flex size-7 items-center justify-center rounded-[7px] bg-zinc-950 text-white" aria-hidden="true">
            <svg className="size-4" viewBox="0 0 16 16" fill="none">
              <rect x="1.5" y="9" width="3.5" height="6" rx="0.8" fill="currentColor" opacity="0.45" />
              <rect x="6.25" y="5" width="3.5" height="10" rx="0.8" fill="currentColor" opacity="0.75" />
              <rect x="11" y="1.5" width="3.5" height="13.5" rx="0.8" fill="currentColor" />
            </svg>
          </span>
          <span>kernelbench</span>
          <span className="text-zinc-400">.com</span>
        </a>
        <div className="hidden items-center gap-7 md:flex">
          <a className="text-sm text-zinc-500 hover:text-zinc-950 no-underline transition" href="/runs">Runs</a>
        </div>
        <div className="flex items-center gap-3">
          <a
            href="https://github.com/hdt98/kernelbench.com"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            className="hidden rounded-full bg-zinc-950 px-5 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 md:inline-flex no-underline"
          >
            GitHub
          </a>
          <a
            href="https://github.com/hdt98/kernelbench.com"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            className="flex size-10 items-center justify-center rounded-lg text-zinc-700 hover:bg-zinc-100 md:hidden no-underline"
          >
            <svg className="size-5" viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.58 2 12.26c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49 0-.24-.01-1.04-.01-1.89-2.78.62-3.37-1.22-3.37-1.22-.45-1.19-1.11-1.5-1.11-1.5-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.9 1.57 2.35 1.12 2.92.86.09-.67.35-1.12.63-1.38-2.22-.26-4.55-1.14-4.55-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05A9.29 9.29 0 0 1 12 6.98c.85 0 1.71.12 2.51.34 1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.81-4.57 5.06.36.32.68.94.68 1.9 0 1.38-.01 2.49-.01 2.83 0 .27.18.59.69.49A10.05 10.05 0 0 0 22 12.26C22 6.58 17.52 2 12 2Z" />
            </svg>
          </a>
        </div>
      </nav>
    </header>
  )
}
