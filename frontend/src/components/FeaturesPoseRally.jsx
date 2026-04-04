import { useId } from 'react'
import { useReducedMotion } from 'framer-motion'

/**
 * BlazePose / MediaPipe-style segment colors (RGB, same grouping as landmark clusters in samples).
 * Left/right are the subject's left/right (mirrored correctly when figure is flipped).
 */
const MP = {
    HEAD_FILL: '#EDE7F6',
    HEAD_STROKE: '#B39DDB',
    TORSO: '#FFEE58',
    LEFT_ARM: '#FF8A65',
    RIGHT_ARM: '#4FC3F7',
    LEFT_LEG: '#AB47BC',
    RIGHT_LEG: '#66BB6A',
    JOINT_HALO: '#FFFFFF',
}

const SHUTTLE_CORK = 'rgba(245, 245, 245, 0.95)'
const SHUTTLE_SKIRT = 'rgba(255, 255, 255, 0.88)'

function connectionStroke(a, b) {
    const s = new Set([a, b])
    if (s.has('hb')) return MP.HEAD_STROKE
    if (s.has('lShoulder') && s.has('rShoulder')) return MP.TORSO
    if (s.has('lHip') && s.has('rHip')) return MP.TORSO
    if (s.has('lShoulder') && s.has('lHip')) return MP.TORSO
    if (s.has('rShoulder') && s.has('rHip')) return MP.TORSO
    if (s.has('lShoulder') && s.has('lElbow')) return MP.LEFT_ARM
    if (s.has('lElbow') && s.has('lWrist')) return MP.LEFT_ARM
    if (s.has('rShoulder') && s.has('rElbow')) return MP.RIGHT_ARM
    if (s.has('rElbow') && s.has('rWrist')) return MP.RIGHT_ARM
    if (s.has('lHip') && s.has('lKnee')) return MP.LEFT_LEG
    if (s.has('lKnee') && s.has('lAnkle')) return MP.LEFT_LEG
    if (s.has('rHip') && s.has('rKnee')) return MP.RIGHT_LEG
    if (s.has('rKnee') && s.has('rAnkle')) return MP.RIGHT_LEG
    return MP.TORSO
}

function jointFill(name) {
    if (name.startsWith('l') && (name.includes('Shoulder') || name.includes('Elbow') || name.includes('Wrist'))) return MP.LEFT_ARM
    if (name.startsWith('r') && (name.includes('Shoulder') || name.includes('Elbow') || name.includes('Wrist'))) return MP.RIGHT_ARM
    if (name.startsWith('l')) return MP.LEFT_LEG
    if (name.startsWith('r')) return MP.RIGHT_LEG
    return MP.TORSO
}

function BadmintonRacket({ wx, wy, tx, ty }) {
    const dx = tx - wx
    const dy = ty - wy
    const len = Math.hypot(dx, dy) || 1
    const ux = dx / len
    const uy = dy / len
    const angleDeg = (Math.atan2(dy, dx) * 180) / Math.PI
    const shaftEx = tx - ux * 11
    const shaftEy = ty - uy * 11
    const headCx = tx + ux * 8
    const headCy = ty + uy * 8
    const mesh = 'rgba(79, 195, 247, 0.2)'

    return (
        <g>
            <line
                x1={wx}
                y1={wy}
                x2={shaftEx}
                y2={shaftEy}
                stroke={MP.RIGHT_ARM}
                strokeWidth={2.8}
                strokeLinecap="round"
            />
            <g transform={`translate(${headCx},${headCy}) rotate(${angleDeg})`}>
                <ellipse cx={0} cy={0} rx={12} ry={5.2} fill={mesh} stroke={MP.RIGHT_ARM} strokeWidth={1.9} />
                <ellipse cx={0} cy={0} rx={8.5} ry={3.2} fill="none" stroke={MP.RIGHT_ARM} strokeWidth={0.9} opacity={0.55} />
                <line x1={-10} y1={0} x2={10} y2={0} stroke={MP.RIGHT_ARM} strokeWidth={0.6} opacity={0.35} />
                <line x1={0} y1={-4} x2={0} y2={4} stroke={MP.RIGHT_ARM} strokeWidth={0.6} opacity={0.35} />
            </g>
        </g>
    )
}

/** Mediapipe-style skeleton facing +x; segment colors match common BlazePose drawing. */
function PoseFigure({ uid }) {
    const head = { cx: 104, cy: 48, r: 12 }
    const headBase = [104, 60]

    const j = {
        lShoulder: [76, 80],
        rShoulder: [132, 80],
        lElbow: [60, 122],
        rElbow: [156, 94],
        lWrist: [52, 166],
        rWrist: [172, 68],
        racketTip: [202, 48],
        lHip: [88, 166],
        rHip: [120, 166],
        lKnee: [82, 220],
        rKnee: [124, 220],
        lAnkle: [78, 274],
        rAnkle: [128, 274],
    }

    const lines = [
        ['hb', 'lShoulder'],
        ['hb', 'rShoulder'],
        ['lShoulder', 'rShoulder'],
        ['lShoulder', 'lElbow'],
        ['lElbow', 'lWrist'],
        ['rShoulder', 'rElbow'],
        ['rElbow', 'rWrist'],
        ['lShoulder', 'lHip'],
        ['rShoulder', 'rHip'],
        ['lHip', 'rHip'],
        ['lHip', 'lKnee'],
        ['lKnee', 'lAnkle'],
        ['rHip', 'rKnee'],
        ['rKnee', 'rAnkle'],
    ]

    const pt = (key) => (key === 'hb' ? headBase : j[key])

    return (
        <g>
            <circle cx={head.cx} cy={head.cy} r={head.r} fill={MP.HEAD_FILL} stroke={MP.HEAD_STROKE} strokeWidth={2} />
            {lines.map(([a, b]) => {
                const p0 = pt(a)
                const p1 = pt(b)
                return (
                    <line
                        key={`${uid}-${a}-${b}`}
                        x1={p0[0]}
                        y1={p0[1]}
                        x2={p1[0]}
                        y2={p1[1]}
                        stroke={connectionStroke(a, b)}
                        strokeWidth={2.1}
                        strokeLinecap="round"
                    />
                )
            })}
            {Object.entries(j)
                .filter(([name]) => name !== 'racketTip')
                .map(([name, [x, y]]) => (
                    <g key={`${uid}-${name}`}>
                        <circle cx={x} cy={y} r={5} fill={MP.JOINT_HALO} opacity={0.95} />
                        <circle cx={x} cy={y} r={3.2} fill={jointFill(name)} />
                    </g>
                ))}
            <BadmintonRacket wx={j.rWrist[0]} wy={j.rWrist[1]} tx={j.racketTip[0]} ty={j.racketTip[1]} />
        </g>
    )
}

function ShuttleGlyph() {
    return (
        <g>
            <ellipse cx={0} cy={2.5} rx={4.5} ry={3} fill={SHUTTLE_CORK} />
            <path d="M0 -1 L5 -8 L0 -4.5 L-5 -8 Z" fill={SHUTTLE_SKIRT} />
        </g>
    )
}

const S = 0.28
const LEFT_TX = 164
const LEFT_TY = 8
const RIGHT_ANCHOR_X = 1035

/**
 * Two pose figures + shuttle; feet align to bottom of SVG (stand on grid card tops). Transparent.
 */
export default function FeaturesPoseRally({ className = '' }) {
    const reduceMotion = useReducedMotion()
    const pathId = `grid-mini-shuttle-${useId().replace(/:/g, '')}`

    const tipLx = LEFT_TX + 202 * S
    const tipLy = LEFT_TY + 48 * S
    const tipRx = RIGHT_ANCHOR_X - 202 * S
    const midX = (tipLx + tipRx) / 2
    const track = `M ${tipLx.toFixed(1)} ${tipLy.toFixed(1)} Q ${midX.toFixed(1)} ${(tipLy - 16).toFixed(1)} ${tipRx.toFixed(1)} ${tipLy.toFixed(1)} Q ${midX.toFixed(1)} ${(tipLy + 16).toFixed(1)} ${tipLx.toFixed(1)} ${tipLy.toFixed(1)}`

    return (
        <svg
            className={className}
            viewBox="0 0 1200 90"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            preserveAspectRatio="xMidYMax meet"
            aria-hidden
        >
            <path id={pathId} d={track} fill="none" stroke="none" />

            <line
                x1={midX}
                y1={4}
                x2={midX}
                y2={86}
                stroke="rgba(255,255,255,0.18)"
                strokeWidth={1}
                strokeDasharray="4 8"
            />

            <g transform={`translate(${LEFT_TX},${LEFT_TY}) scale(${S})`}>
                <PoseFigure uid="L" />
            </g>
            <g transform={`translate(${RIGHT_ANCHOR_X},${LEFT_TY}) scale(${-S}, ${S})`}>
                <PoseFigure uid="R" />
            </g>

            {reduceMotion ? (
                <g transform={`translate(${midX}, ${tipLy - 5})`}>
                    <ShuttleGlyph />
                </g>
            ) : (
                <g>
                    <animateMotion dur="5s" repeatCount="indefinite" rotate="auto">
                        <mpath href={`#${pathId}`} />
                    </animateMotion>
                    <ShuttleGlyph />
                </g>
            )}
        </svg>
    )
}
