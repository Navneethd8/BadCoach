import { useNavigate } from 'react-router-dom'
import ThemeToggle from './ThemeToggle.jsx'

export default function LegalLayout({ title, children }) {
    const navigate = useNavigate()
    return (
        <div className="min-h-screen bg-page font-sans text-foreground px-6 py-12 md:px-10 md:py-16">
            <div className="mx-auto mb-8 flex max-w-2xl justify-end md:mb-10">
                <ThemeToggle />
            </div>
            <div className="mx-auto max-w-2xl border-l-2 border-brand pl-8 md:pl-12">
                <button
                    type="button"
                    onClick={() => navigate(-1)}
                    className="font-mono text-[11px] uppercase tracking-[0.22em] text-brand transition-colors hover:text-accent"
                >
                    ← Back
                </button>
                <h1 className="font-display mt-10 text-4xl font-normal tracking-tight text-foreground md:text-5xl">{title}</h1>
                <div className="mt-12 space-y-6 text-base leading-relaxed text-foreground-muted">{children}</div>
            </div>
        </div>
    )
}
