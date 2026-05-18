import { Link } from 'react-router-dom'
import Logo from './Logo'

export default function LegalLayout({ title, children }) {
    return (
        <div className="min-h-screen bg-[#fafafa] text-neutral-900">
            <header className="border-b border-neutral-200 bg-[#6c9c8d] px-5 py-4">
                <div className="mx-auto flex max-w-2xl items-center justify-between gap-4">
                    <Link to="/" className="flex items-center gap-2 text-[#fafafa]" aria-label="IsoCourt home">
                        <Logo size={22} className="shrink-0 text-[#fafafa]" />
                        <span className="font-display text-sm font-bold tracking-tight">IsoCourt</span>
                    </Link>
                </div>
            </header>
            <main className="px-5 py-10 sm:px-8">
                <div className="mx-auto max-w-2xl">
                    <Link to="/" className="figma-legal-back">
                        ← Back to home
                    </Link>
                    <h1 className="figma-section-title mt-6 text-2xl sm:text-3xl">{title.toLowerCase()}</h1>
                    <div className="mt-8 space-y-4 text-sm text-neutral-700 leading-relaxed font-sans">{children}</div>
                </div>
            </main>
        </div>
    )
}
