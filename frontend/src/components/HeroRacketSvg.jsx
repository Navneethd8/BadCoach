/** Pure vector racket — Figma 216:366 paths (no raster images). */

/* Head interior ellipse — clips the string mesh to the inside of the head frame.
   Inner head bounds (from the head's outer-minus-inner path) ≈
   x [56.96, 1394.18], y [45.73, 1119.43]
   center (725.57, 582.58), radii ≈ (668.6, 536.85).
   Strings run flush to the inner ring (no white rim). */
const HEAD_CX = 725.57
const HEAD_CY = 582.58
const HEAD_RX = 668.6
const HEAD_RY = 536.85

const STRING_COLOR = '#1B4D3E'
const STRING_WIDTH = 3

/* Tight mesh — smaller cells so the strings read as a real string bed. */
const VERTICAL_COUNT = 24
const HORIZONTAL_COUNT = 20

const verticals = Array.from({ length: VERTICAL_COUNT }, (_, i) => {
    const x = HEAD_CX - HEAD_RX + ((2 * HEAD_RX) / (VERTICAL_COUNT + 1)) * (i + 1)
    return (
        <line
            key={`v${i}`}
            x1={x}
            y1={HEAD_CY - HEAD_RY - 40}
            x2={x}
            y2={HEAD_CY + HEAD_RY + 40}
            stroke={STRING_COLOR}
            strokeWidth={STRING_WIDTH}
        />
    )
})

const horizontals = Array.from({ length: HORIZONTAL_COUNT }, (_, i) => {
    const y = HEAD_CY - HEAD_RY + ((2 * HEAD_RY) / (HORIZONTAL_COUNT + 1)) * (i + 1)
    return (
        <line
            key={`h${i}`}
            x1={HEAD_CX - HEAD_RX - 40}
            y1={y}
            x2={HEAD_CX + HEAD_RX + 40}
            y2={y}
            stroke={STRING_COLOR}
            strokeWidth={STRING_WIDTH}
        />
    )
})

export default function HeroRacketSvg() {
    return (
        <svg
            className="figma-racket-svg"
            viewBox="0 0 3861.12 1165.17"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden
        >
            <defs>
                <clipPath id="iso-racket-head-clip">
                    <ellipse cx={HEAD_CX} cy={HEAD_CY} rx={HEAD_RX} ry={HEAD_RY} />
                </clipPath>

                {/* Radial fade: strings invisible at the center (clear for the
                    title + CTAs) and ramp to full visibility near the frame.
                    Luminance mode: black = invisible, white = visible. */}
                <radialGradient
                    id="iso-racket-strings-fade"
                    cx={HEAD_CX}
                    cy={HEAD_CY}
                    r={HEAD_RX}
                    gradientUnits="userSpaceOnUse"
                >
                    <stop offset="0%" stopColor="#000" />
                    <stop offset="55%" stopColor="#000" />
                    <stop offset="85%" stopColor="#b3b3b3" />
                    <stop offset="100%" stopColor="#fff" />
                </radialGradient>

                <mask id="iso-racket-strings-mask" maskUnits="userSpaceOnUse">
                    <rect
                        x="0"
                        y="0"
                        width="3861.12"
                        height="1165.17"
                        fill="url(#iso-racket-strings-fade)"
                    />
                </mask>
            </defs>

            {/* Solid page-color fill inside the head — occludes the page-wide
                net pattern so only the string mesh shows through the racket face. */}
            <ellipse
                cx={HEAD_CX}
                cy={HEAD_CY}
                rx={HEAD_RX}
                ry={HEAD_RY}
                fill="#fafafa"
            />

            {/* String mesh — clipped to the head, masked so the center is clear
                for the copy and the strings only ramp in near the frame. */}
            <g
                className="figma-racket-strings"
                clipPath="url(#iso-racket-head-clip)"
                mask="url(#iso-racket-strings-mask)"
            >
                {verticals}
                {horizontals}
            </g>

            <g className="figma-racket-shaft">
                <path
                    d="M1451.14 582.583C1451.14 904.335 1126.29 1165.17 725.571 1165.17C324.849 1165.17 0.000137329 904.335 0.000137329 582.583C0.000137329 260.831 324.849 0 725.571 0C1126.29 0 1451.14 260.831 1451.14 582.583ZM56.9577 582.583C56.9577 879.077 356.306 1119.43 725.571 1119.43C1094.84 1119.43 1394.18 879.077 1394.18 582.583C1394.18 286.089 1094.84 45.7329 725.571 45.7329C356.306 45.7329 56.9577 286.089 56.9577 582.583Z"
                    fill="#000"
                />
                <path
                    d="M1266.05 258.662C1343.47 367.687 1407.97 450.879 1359.2 556.275H1399.97C1429.4 425.599 1369.49 347.595 1266.05 258.662Z"
                    fill="#000"
                    stroke="#000"
                    strokeWidth="0.791016"
                />
                <path
                    d="M1272.85 895.43C1350.27 786.405 1407.57 661.48 1358.8 556.084H1399.57C1429 686.76 1376.29 806.497 1272.85 895.43Z"
                    fill="#000"
                    stroke="#000"
                    strokeWidth="0.791016"
                />
            </g>
            <g className="figma-racket-head" transform="translate(1418.96 412.12)">
                <path
                    d="M2359.68 0C2499.61 40.5318 2434.67 195.867 2359.68 198.545C2284.69 201.223 2119.83 196.081 1774.1 198.545C1700.29 199.071 1699.5 198.545 1531.8 180.352L1293.31 157.412L193.403 170.859C161.757 170.957 112.72 187.075 82.6611 210.015C79.3616 211.818 3.20958 285.618 0 287.534C1.64473 280.687 3.7428 264.145 5.80664 244.501L104.414 145.547L5.80176 50.7539C3.66735 32.5221 1.53349 16.9337 0 9.8877C2.42515 11.4515 80.1821 89.4635 82.6611 90.9668C101.112 102.144 111.138 113.511 139.219 113.511C164.136 113.511 176.813 113.464 193.403 113.511C665.64 101.25 822.028 120.376 1293.31 106.787L1294.6 106.75L1774.1 42.4092C2052.12 36.0811 2238.93 0.402848 2359.68 0Z"
                    fill="#000"
                />
            </g>
        </svg>
    )
}
