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
/* Outer rim of the head ring (donut path uses these for the exterior ellipse). */
const RIM_RX = 725.571
const RIM_RY = 582.583
/* Padding so dark-mode rim stroke is not clipped by the viewBox. */
const VB_PAD = 18
const VB_W = 3861.12
const VB_H = 1165.17

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
            stroke="var(--racket-string)"
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
            stroke="var(--racket-string)"
            strokeWidth={STRING_WIDTH}
        />
    )
})

export default function HeroRacketSvg() {
    return (
        <svg
            className="figma-racket-svg"
            viewBox={`${-VB_PAD} ${-VB_PAD} ${VB_W + VB_PAD * 2} ${VB_H + VB_PAD * 2}`}
            fill="none"
            overflow="visible"
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
                className="figma-racket-head-fill"
                cx={HEAD_CX}
                cy={HEAD_CY}
                rx={HEAD_RX}
                ry={HEAD_RY}
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

            {/* Dark-mode outer rim only — avoids inner edge stroke on the donut path */}
            <ellipse
                className="figma-racket-outer-rim"
                cx={HEAD_CX}
                cy={HEAD_CY}
                rx={RIM_RX}
                ry={RIM_RY}
            />

            <g className="figma-racket-shaft">
                <path
                    className="figma-racket-oval"
                    d="M1451.14 582.58C1451.14 904.34 1126.29 1165.17 725.57 1165.17C324.85 1165.17 0 904.34 0 582.58C0 260.83 324.85 0 725.57 0C1126.29 0 1451.14 260.83 1451.14 582.58ZM56.96 582.58C56.96 879.08 356.31 1119.43 725.57 1119.43C1094.84 1119.43 1394.18 879.08 1394.18 582.58C1394.18 286.09 1094.84 45.73 725.57 45.73C356.31 45.73 56.96 286.09 56.96 582.58Z"
                    fill="var(--racket-frame)"
                />
                {/* Decorative curves inside the head — fill only, no rim stroke */}
                <path
                    className="figma-racket-head-inner-curve"
                    d="M1266.05 258.66C1343.47 367.69 1407.97 450.88 1359.2 556.27H1399.97C1429.4 425.6 1369.49 347.6 1266.05 258.66Z"
                    fill="var(--racket-frame)"
                />
                <path
                    className="figma-racket-head-inner-curve"
                    d="M1272.85 895.43C1350.27 786.4 1407.57 661.48 1358.8 556.08H1399.57C1429 686.76 1376.29 806.5 1272.85 895.43Z"
                    fill="var(--racket-frame)"
                />
            </g>
            <g className="figma-racket-grip" transform="translate(1418.96 412.12)">
                <path
                    className="figma-racket-grip-outline"
                    d="M2359.68 0C2499.61 40.53 2434.67 195.87 2359.68 198.54C2284.69 201.22 2119.83 196.08 1774.1 198.54C1700.29 199.07 1699.5 198.54 1531.8 180.35L1293.31 157.41L193.4 170.86C161.76 170.96 112.72 187.07 82.66 210.01C79.36 211.82 3.21 285.62 0 287.53C1.64 280.69 3.74 264.14 5.81 244.5L104.41 145.55L5.8 50.75C3.67 32.52 1.53 16.93 0 9.89C2.43 11.45 80.18 89.46 82.66 90.97C101.11 102.14 111.14 113.51 139.22 113.51C164.14 113.51 176.81 113.46 193.4 113.51C665.64 101.25 822.03 120.38 1293.31 106.79L1294.6 106.75L1774.1 42.41C2052.12 36.08 2238.93 0.4 2359.68 0Z"
                    fill="var(--racket-frame)"
                />
            </g>
        </svg>
    )
}
