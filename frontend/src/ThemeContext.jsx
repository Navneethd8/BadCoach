import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useState } from 'react'

const STORAGE_KEY = 'isocourt-theme'

function applyDom(mode) {
    document.documentElement.classList.toggle('dark', mode === 'dark')
}

function readStoredMode() {
    if (typeof window === 'undefined') return null
    try {
        const s = localStorage.getItem(STORAGE_KEY)
        if (s === 'light' || s === 'dark') return s
    } catch {
        /* ignore */
    }
    return null
}

function getInitialMode() {
    const stored = readStoredMode()
    if (stored) return stored
    if (typeof window === 'undefined') return 'light'
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

const ThemeContext = createContext(null)

export function ThemeProvider({ children }) {
    const [mode, setModeState] = useState(getInitialMode)

    useLayoutEffect(() => {
        applyDom(mode)
    }, [mode])

    useLayoutEffect(() => {
        const mq = window.matchMedia('(prefers-color-scheme: dark)')
        const onChange = () => {
            if (readStoredMode()) return
            const next = mq.matches ? 'dark' : 'light'
            setModeState(next)
            applyDom(next)
        }
        mq.addEventListener('change', onChange)
        return () => mq.removeEventListener('change', onChange)
    }, [])

    const setMode = useCallback((next) => {
        try {
            localStorage.setItem(STORAGE_KEY, next)
        } catch {
            /* ignore */
        }
        setModeState(next)
        applyDom(next)
    }, [])

    const toggle = useCallback(() => {
        setModeState((m) => {
            const next = m === 'dark' ? 'light' : 'dark'
            try {
                localStorage.setItem(STORAGE_KEY, next)
            } catch {
                /* ignore */
            }
            applyDom(next)
            return next
        })
    }, [])

    const value = useMemo(() => ({ mode, setMode, toggle }), [mode, setMode, toggle])

    return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
    const ctx = useContext(ThemeContext)
    if (!ctx) throw new Error('useTheme must be used within ThemeProvider')
    return ctx
}
