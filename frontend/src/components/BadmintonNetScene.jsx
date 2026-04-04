import { useId } from 'react'

/**
 * Minimalist badminton net in perspective: black court, white mesh,
 * crimson tape and posts. Decorative only (aria-hidden).
 */

const NET = {
    tl: [120, 158],
    tr: [682, 118],
    br: [724, 382],
    bl: [48, 422],
}

const RED = '#8B0000'
const MESH = '#FFFFFF'

function lerp(a, b, t) {
    return a + (b - a) * t
}

function edgePoint(p0, p1, t) {
    return [lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t)]
}

function buildGridPath(cols, rows) {
    const { tl, tr, br, bl } = NET
    const parts = []
    for (let i = 0; i <= rows; i++) {
        const v = i / rows
        const left = edgePoint(bl, tl, v)
        const right = edgePoint(br, tr, v)
        parts.push(`M${left[0].toFixed(1)},${left[1].toFixed(1)}L${right[0].toFixed(1)},${right[1].toFixed(1)}`)
    }
    for (let j = 0; j <= cols; j++) {
        const u = j / cols
        const bottom = edgePoint(bl, br, u)
        const top = edgePoint(tl, tr, u)
        parts.push(`M${bottom[0].toFixed(1)},${bottom[1].toFixed(1)}L${top[0].toFixed(1)},${top[1].toFixed(1)}`)
    }
    return parts.join('')
}

const gridPath = buildGridPath(14, 6)

export default function BadmintonNetScene({ className = '' }) {
    const { tl, tr, br, bl } = NET
    const clipId = `badminton-net-clip-${useId().replace(/:/g, '')}`
    return (
        <svg
            className={className}
            viewBox="0 0 800 500"
            preserveAspectRatio="xMidYMid slice"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden
        >
            <defs>
                <clipPath id={clipId}>
                    <polygon points={`${tl.join(',')} ${tr.join(',')} ${br.join(',')} ${bl.join(',')}`} />
                </clipPath>
            </defs>

            <rect width="800" height="500" fill="#000000" />

            <g clipPath={`url(#${clipId})`} className="badminton-net-mesh">
                <path d={gridPath} fill="none" stroke={MESH} strokeWidth={0.85} vectorEffect="non-scaling-stroke" />
            </g>

            <path
                d={`M${tl[0]},${tl[1]} L${tr[0]},${tr[1]}`}
                fill="none"
                stroke={RED}
                strokeWidth={14}
                strokeLinecap="butt"
                strokeLinejoin="miter"
            />
            <path
                d={`M${bl[0]},${bl[1]} L${tl[0]},${tl[1]}`}
                fill="none"
                stroke={RED}
                strokeWidth={12}
                strokeLinecap="square"
            />
            <path
                d={`M${br[0]},${br[1]} L${tr[0]},${tr[1]}`}
                fill="none"
                stroke={RED}
                strokeWidth={12}
                strokeLinecap="square"
            />
        </svg>
    )
}
