import { useTheme } from '../context/ThemeContext'
import SvgIcon from './SvgIcon'

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
            <SvgIcon name={isDark ? 'light_mode' : 'dark_mode'} size={18} />
        </button>
    )
}
