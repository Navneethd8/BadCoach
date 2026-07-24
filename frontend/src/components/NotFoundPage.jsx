import { Link } from 'react-router-dom'
import { usePageSeo } from '../seo/usePageSeo'
import Logo from './Logo'
import ThemeToggle from './ThemeToggle'

export default function NotFoundPage() {
    usePageSeo('/404', { noindex: true })

    return (
        <div className="theme-page min-h-screen">
            <header className="figma-top-bar">
                <div className="mx-auto flex h-14 sm:h-16 max-w-2xl items-center justify-between gap-4 px-5 sm:px-8">
                    <Link to="/" className="flex items-center gap-2 text-white" aria-label="IsoCourt home">
                        <Logo size={22} className="shrink-0 text-white" />
                        <span className="font-display text-sm font-bold tracking-tight">IsoCourt</span>
                    </Link>
                    <ThemeToggle />
                </div>
            </header>
            <div className="figma-top-bar-spacer" aria-hidden />
            <main className="px-5 py-16 sm:px-8">
                <div className="mx-auto max-w-2xl text-center">
                    <p className="font-mono text-xs uppercase tracking-[0.2em] text-[var(--text-muted)] mb-4">404</p>
                    <h1 className="figma-section-title text-2xl sm:text-3xl">page not found</h1>
                    <p className="mt-4 text-sm text-[var(--text-secondary)] leading-relaxed">
                        That URL is not part of IsoCourt. Head home or open Analyze to drop a clip.
                    </p>
                    <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                        <Link to="/" className="figma-cta figma-cta--primary">
                            Home
                        </Link>
                        <Link to="/analyze" className="figma-cta figma-cta--secondary">
                            Analyze a clip
                        </Link>
                    </div>
                </div>
            </main>
        </div>
    )
}
