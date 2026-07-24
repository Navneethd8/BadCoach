/** Lightweight inline icons for marketing surfaces (avoids 1.5MB Material Symbols on LCP). */

const PATHS = {
    directions_run: 'M13.5 5.5a1.5 1.5 0 1 0-3 0 1.5 1.5 0 0 0 3 0ZM9.8 8.9l-1.5 2.4-2.1-.7-.7 1.9 3 .9 1.1 1.8-.9 4.2 2 .4.9-3.6 1.4 1.1.5 3.4 2-.3-.6-4.2-2.3-1.9 1.3-2.1 1.4.5.7-1.9-3.1-1Zm2.7 3.2-1.4-2.2 1.7-.2 1.3 2.1-1.6.3Z',
    query_stats: 'M5 19V9h2v10H5Zm6 0V5h2v14h-2Zm6 0v-6h2v6h-2Z',
    tips_and_updates: 'M12 2a7 7 0 0 0-4 12.7V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.3A7 7 0 0 0 12 2Zm-2 17h4v1a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1v-1Z',
    check_circle: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm-1.2 13.3-3.1-3.1 1.4-1.4 1.7 1.7 4.2-4.2 1.4 1.4-5.6 5.6Z',
    pan_tool_alt: 'M10 2a1 1 0 0 1 1 1v7.2l1.2-.8a2 2 0 0 1 2.8.4l.4.6 1.5-1a2 2 0 0 1 2.7.6l.2.3A4 4 0 0 1 16 17H9a4 4 0 0 1-4-4V8a1 1 0 0 1 2 0v4h1V3a1 1 0 0 1 1-1Z',
    explore: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm3.9 6.1-2.2 5.3-5.3 2.2 2.2-5.3 5.3-2.2Z',
    location_on: 'M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7Zm0 9.5A2.5 2.5 0 1 1 12 6a2.5 2.5 0 0 1 0 5.5Z',
    psychology: 'M13 2a6 6 0 0 0-5.7 8H6a3 3 0 0 0 0 6h1.1A5 5 0 0 0 11 20h2a1 1 0 0 0 1-1v-2.1A6 6 0 0 0 13 2Zm-1 9a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3Z',
    dark_mode: 'M12.5 3a8.5 8.5 0 1 0 8.4 9.5A7 7 0 0 1 12.5 3Z',
    light_mode: 'M12 4.5a1 1 0 0 1 1 1V7a1 1 0 1 1-2 0V5.5a1 1 0 0 1 1-1Zm0 11a1 1 0 0 1 1 1V18a1 1 0 1 1-2 0v-1.5a1 1 0 0 1 1-1ZM5.5 11a1 1 0 0 1 1-1H8a1 1 0 1 1 0 2H6.5a1 1 0 0 1-1-1Zm11 0a1 1 0 0 1 1-1H19a1 1 0 1 1 0 2h-1.5a1 1 0 0 1-1-1ZM6.7 6.7a1 1 0 0 1 1.4 0l1 1a1 1 0 0 1-1.4 1.4l-1-1a1 1 0 0 1 0-1.4Zm8.2 8.2a1 1 0 0 1 1.4 0l1 1a1 1 0 1 1-1.4 1.4l-1-1a1 1 0 0 1 0-1.4ZM17.3 6.7a1 1 0 0 1 0 1.4l-1 1a1 1 0 1 1-1.4-1.4l1-1a1 1 0 0 1 1.4 0Zm-8.2 8.2a1 1 0 0 1 0 1.4l-1 1a1 1 0 1 1-1.4-1.4l1-1a1 1 0 0 1 1.4 0ZM12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8Z',
}

export default function SvgIcon({ name, size = 20, className = '' }) {
    const d = PATHS[name]
    if (!d) return null
    return (
        <svg
            className={className}
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden
            focusable="false"
        >
            <path d={d} />
        </svg>
    )
}
