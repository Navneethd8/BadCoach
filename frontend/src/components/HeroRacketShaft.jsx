/**
 * Shaft + grip + neck are fixed. Only the head rim (ellipses) rotates to meet the neck.
 */
export default function HeroRacketShaft({ className = '' }) {
    const vbW = 1200

    const RACKET_MM = 680
    const HEAD_W_MM = 280
    const HEAD_L_MM = 220
    const GRIP_MM = 100
    const CONE_MM = 28

    const s = vbW / RACKET_MM

    const gripW = GRIP_MM * s
    const gripX2 = vbW
    const gripX = gripX2 - gripW
    const coneW = CONE_MM * s
    const coneLeft = gripX - coneW

    const headRx = (HEAD_W_MM / 2) * s
    const headRy = (HEAD_L_MM / 2) * s
    const frameInRx = headRx * 0.9
    const frameInRy = headRy * 0.86

    const shaftH = 22
    const gripH = 50
    const padTop = 18
    const padBottom = 22
    const padLeft = 140
    const padSky = 50

    const vbH = padTop + headRy * 2 + padBottom
    const rowY = padTop + headRy

    const shaftTop = rowY - shaftH / 2
    const shaftBottom = rowY + shaftH / 2
    const gripTop = rowY - gripH / 2
    const gripBottom = rowY + gripH / 2

    const headCx = -200
    const neckX = headCx + headRx * 0.9
    const throatX = neckX + 6
    const headTiltDeg = 0
    const neckHalf = shaftH * 0.45

    return (
        <svg
            viewBox={`${-padLeft} ${-padSky} ${vbW + padLeft} ${vbH + padSky}`}
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className={className}
            aria-hidden
            preserveAspectRatio="xMaxYMax slice"
        >
            {/* Locked: neck + shaft + grip */}
            <path
                fill="currentColor"
                d={`
                    M ${neckX} ${rowY - neckHalf}
                    L ${throatX} ${shaftTop}
                    L ${throatX} ${shaftBottom}
                    L ${neckX} ${rowY + neckHalf}
                    Z
                `}
            />
            <path
                fill="currentColor"
                d={`
                    M ${gripX2} ${gripBottom}
                    L ${gripX} ${gripBottom}
                    L ${coneLeft} ${shaftBottom}
                    L ${throatX} ${shaftBottom}
                    L ${throatX} ${shaftTop}
                    L ${coneLeft} ${shaftTop}
                    L ${gripX} ${gripTop}
                    L ${gripX2} ${gripTop}
                    Z
                `}
            />

            {/* Rim only — rotates to meet fixed neck at (neckX, rowY) */}
            <g transform={`rotate(${headTiltDeg}, ${neckX}, ${rowY})`}>
                <ellipse cx={headCx} cy={rowY} rx={headRx} ry={headRy} fill="currentColor" />
                <ellipse cx={headCx} cy={rowY} rx={frameInRx} ry={frameInRy} fill="#0a0a0a" />
            </g>
        </svg>
    )
}
