/** @type {import('tailwindcss').Config} */
export default {
    darkMode: 'class',
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                brand: {
                    DEFAULT: '#059669',
                    dark: '#047857',
                    accent: '#047857',
                },
                page: '#fafafa',
            },
            fontFamily: {
                display: ['"Iosevka Charon Mono"', 'ui-monospace', 'monospace'],
                mono: ['"Iosevka Charon Mono"', 'ui-monospace', 'monospace'],
            },
        },
    },
    plugins: [],
}
