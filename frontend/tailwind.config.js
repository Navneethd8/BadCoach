/** @type {import('tailwindcss').Config} */
export default {
    darkMode: 'class',
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                display: ['Fraunces', 'Georgia', 'ui-serif', 'serif'],
                sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                mono: ['IBM Plex Mono', 'ui-monospace', 'monospace'],
            },
            colors: {
                page: {
                    DEFAULT: 'var(--color-page)',
                    muted: 'var(--color-page-muted)',
                },
                surface: {
                    DEFAULT: 'var(--color-surface)',
                    elevated: 'var(--color-surface-elevated)',
                },
                foreground: {
                    DEFAULT: 'var(--color-text)',
                    muted: 'var(--color-text-muted)',
                    subtle: 'var(--color-text-subtle)',
                },
                border: {
                    DEFAULT: 'var(--color-border)',
                    strong: 'var(--color-border-strong)',
                },
                muted: 'var(--color-muted)',
                inset: 'var(--color-inset)',
                overlay: 'var(--color-overlay)',
                accent: {
                    DEFAULT: 'var(--color-accent)',
                    hover: 'var(--color-accent-hover)',
                    muted: 'var(--color-accent-muted)',
                },
                onaccent: 'var(--color-on-accent)',
                brand: {
                    DEFAULT: 'var(--color-brand)',
                    secondary: 'var(--color-brand-secondary)',
                },
                success: 'var(--color-success)',
                warning: 'var(--color-warning)',
                danger: 'var(--color-danger)',
                coach: {
                    tint: 'var(--color-coach-tint)',
                    border: 'var(--color-coach-border)',
                    fg: 'var(--color-coach-text)',
                },
                court: {
                    mat: 'var(--court-mat)',
                    deep: 'var(--court-mat-deep)',
                    line: 'var(--court-line)',
                    on: 'var(--court-on)',
                },
            },
        },
    },
    plugins: [],
}
