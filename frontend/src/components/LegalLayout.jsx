import { Link } from 'react-router-dom'
import Logo from './Logo'
import ThemeToggle from './ThemeToggle'

/**
 * Shared chrome for legal / learn pages.
 * `hero` uses the same mono title treatment as Analyze / Live.
 */
export default function LegalLayout({ title, lead, hero = false, children }) {
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
            <main className="px-5 py-10 sm:px-8">
                <div className="mx-auto max-w-2xl">
                    <Link to="/" className="figma-legal-back">
                        ← Back to home
                    </Link>
                    <h1
                        className={
                            hero
                                ? 'app-page-title mt-6'
                                : 'figma-section-title mt-6 text-2xl sm:text-3xl'
                        }
                    >
                        {title}
                    </h1>
                    {lead ? <p className="app-page-lead mb-2">{lead}</p> : null}
                    <div className="mt-8 space-y-4 text-sm text-[var(--text-secondary)] leading-relaxed font-sans">{children}</div>
                    <nav className="mt-12 flex flex-wrap gap-x-5 gap-y-2 text-xs text-[var(--text-muted)]" aria-label="Learn">
                        <Link to="/faq" className="hover:text-brand transition-colors">FAQ</Link>
                        <Link to="/glossary" className="hover:text-brand transition-colors">Glossary</Link>
                        <Link to="/what-is-ai-badminton-stroke-analysis" className="hover:text-brand transition-colors">What is</Link>
                        <Link to="/compare" className="hover:text-brand transition-colors">Compare</Link>
                        <Link to="/privacy" className="hover:text-brand transition-colors">Privacy</Link>
                        <Link to="/terms" className="hover:text-brand transition-colors">Terms</Link>
                    </nav>
                </div>
            </main>
        </div>
    )
}
