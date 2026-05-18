/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                brand: {
                    DEFAULT: '#6c9c8d',
                    dark: '#5a8578',
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
