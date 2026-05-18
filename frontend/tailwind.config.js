/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                brand: '#6c9c8d',
                page: '#fafafa',
            },
            fontFamily: {
                display: ['"JetBrains Mono"', '"IBM Plex Mono"', 'ui-monospace', 'monospace'],
                mono: ['"JetBrains Mono"', '"IBM Plex Mono"', 'ui-monospace', 'monospace'],
            },
        },
    },
    plugins: [],
}
