import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

const STORAGE_KEY = 'isocourt-theme'

const ThemeContext = createContext(null)

function getSystemTheme() {
    if (typeof window === 'undefined') return 'light'
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function readStoredTheme() {
    if (typeof window === 'undefined') return null
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored === 'light' || stored === 'dark' ? stored : null
}

function applyTheme(theme) {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    document.documentElement.style.colorScheme = theme
}

export function ThemeProvider({ children }) {
    const [theme, setThemeState] = useState(() => readStoredTheme() ?? getSystemTheme())

    useEffect(() => {
        applyTheme(theme)
        localStorage.setItem(STORAGE_KEY, theme)
    }, [theme])

    useEffect(() => {
        const stored = readStoredTheme()
        if (stored) return

        const mq = window.matchMedia('(prefers-color-scheme: dark)')
        const onChange = () => {
            if (!readStoredTheme()) setThemeState(mq.matches ? 'dark' : 'light')
        }
        mq.addEventListener('change', onChange)
        return () => mq.removeEventListener('change', onChange)
    }, [])

    const setTheme = useCallback((next) => {
        setThemeState(next === 'dark' ? 'dark' : 'light')
    }, [])

    const toggleTheme = useCallback(() => {
        setThemeState((t) => (t === 'dark' ? 'light' : 'dark'))
    }, [])

    const value = useMemo(
        () => ({
            theme,
            resolvedTheme: theme,
            setTheme,
            toggleTheme,
            isDark: theme === 'dark',
        }),
        [theme, setTheme, toggleTheme],
    )

    return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
    const ctx = useContext(ThemeContext)
    if (!ctx) throw new Error('useTheme must be used within ThemeProvider')
    return ctx
}
