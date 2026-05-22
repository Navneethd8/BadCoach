import { Link } from 'react-router-dom'
import Logo from './Logo'
import ThemeToggle from './ThemeToggle'

export default function AppShell({ active, children, mainClassName = 'mx-auto max-w-2xl px-5 py-8 sm:px-8' }) {
    return (
        <>
            <header className="figma-top-bar">
                <div className="mx-auto flex h-14 sm:h-16 max-w-6xl items-center justify-between gap-4 px-5 sm:px-8">
                    <Link to="/" className="flex min-w-0 items-center gap-2.5 text-[#fafafa]" aria-label="IsoCourt home">
                        <Logo size={24} className="shrink-0 text-[#fafafa]" />
                        <span className="font-display text-lg font-bold tracking-tight hidden sm:inline">IsoCourt</span>
                    </Link>
                    <nav className="flex items-center gap-3 sm:gap-5" aria-label="Primary">
                        <Link
                            to="/analyze"
                            className={`app-nav-link ${active === 'analyze' ? 'app-nav-link--active' : ''}`}
                        >
                            Analyze
                        </Link>
                        <Link
                            to="/live"
                            className={`app-nav-link ${active === 'live' ? 'app-nav-link--active' : ''}`}
                        >
                            Live
                        </Link>
                        <ThemeToggle />
                    </nav>
                </div>
            </header>

            <div className="app-page figma-page-body theme-page min-h-screen w-full">
                <div className="figma-top-bar-spacer" aria-hidden />
                <main className={mainClassName}>{children}</main>
            </div>
        </>
    )
}
