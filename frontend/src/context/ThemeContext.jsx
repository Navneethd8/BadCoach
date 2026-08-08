import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

const STORAGE_KEY = 'isocourt-theme'

const ThemeContext = createContext(null)

function readStoredTheme() {
    if (typeof window === 'undefined') return null
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored === 'light' || stored === 'dark' ? stored : null
}

function preferDark() {
    if (typeof window === 'undefined') return false
    return window.matchMedia('(prefers-color-scheme: dark)').matches
}

function applyTheme(theme) {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    document.documentElement.style.colorScheme = theme
}

export function ThemeProvider({ children }) {
    // Match index.html FOUC script — don't snap dark OS users to light on first paint.
    const [theme, setThemeState] = useState(() => readStoredTheme() ?? (preferDark() ? 'dark' : 'light'))

    useEffect(() => {
        applyTheme(theme)
        localStorage.setItem(STORAGE_KEY, theme)
    }, [theme])

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
