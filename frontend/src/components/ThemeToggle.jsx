import { useTheme } from '../context/ThemeContext'

export default function ThemeToggle({ className = '' }) {
    const { isDark, toggleTheme } = useTheme()

    return (
        <button
            type="button"
            onClick={toggleTheme}
            className={`theme-toggle ${className}`.trim()}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            title={isDark ? 'Light mode' : 'Dark mode'}
        >
            <span className="material-symbols-outlined" style={{ fontSize: 18 }} aria-hidden>
                {isDark ? 'light_mode' : 'dark_mode'}
            </span>
        </button>
    )
}
