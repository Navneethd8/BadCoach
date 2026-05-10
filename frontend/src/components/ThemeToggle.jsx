import { useTheme } from '../ThemeContext.jsx'

/** Square control — matches Sunrise Court chrome (not a SaaS pill). */
export default function ThemeToggle({ className = '' }) {
    const { mode, toggle } = useTheme()

    return (
        <button
            type="button"
            onClick={toggle}
            aria-label={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={mode === 'dark' ? 'Light mode' : 'Dark mode'}
            className={`
                inline-flex h-9 w-9 shrink-0 items-center justify-center
                border-2 border-border bg-page-muted text-foreground-muted
                transition-colors hover:border-brand hover:text-foreground
                rounded-none
                ${className}
            `.trim().replace(/\s+/g, ' ')}
        >
            <span className="material-symbols-outlined text-[20px]" aria-hidden>
                {mode === 'dark' ? 'light_mode' : 'dark_mode'}
            </span>
        </button>
    )
}
