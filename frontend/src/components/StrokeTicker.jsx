import { useEffect, useRef, useState } from 'react'

const STROKES = [
    'smash',
    'overhead clear',
    'net drop',
    'drive',
    'lift',
    'backhand push',
    'cross-court drop',
    'jump smash',
]

const MIN_COPIES = 2
const MAX_COPIES = 14

function TickerRow() {
    return (
        <>
            {STROKES.map((label, i) => (
                <span key={i} className="figma-stroke-ticker__label">
                    {label}
                    {i < STROKES.length - 1 ? (
                        <span className="figma-stroke-ticker__sep" aria-hidden>
                            ·
                        </span>
                    ) : null}
                </span>
            ))}
        </>
    )
}

export default function StrokeTicker() {
    const viewportRef = useRef(null)
    const measureRef = useRef(null)
    const [copies, setCopies] = useState(MIN_COPIES)

    useEffect(() => {
        const viewport = viewportRef.current
        const row = measureRef.current
        if (!viewport || !row) return

        const updateCopies = () => {
            const viewWidth = viewport.offsetWidth
            const rowWidth = row.offsetWidth
            if (viewWidth <= 0 || rowWidth <= 0) return

            /* Enough repeats so one row width of scroll never exposes empty viewport. */
            const needed = Math.min(
                MAX_COPIES,
                Math.max(MIN_COPIES, Math.ceil((viewWidth * 2) / rowWidth)),
            )
            setCopies((prev) => (prev === needed ? prev : needed))
        }

        updateCopies()
        const observer = new ResizeObserver(updateCopies)
        observer.observe(viewport)
        observer.observe(row)
        window.addEventListener('resize', updateCopies)

        return () => {
            observer.disconnect()
            window.removeEventListener('resize', updateCopies)
        }
    }, [])

    return (
        <div className="figma-stroke-ticker" aria-hidden>
            <div ref={viewportRef} className="figma-stroke-ticker__viewport">
                <div
                    className="figma-stroke-ticker__track"
                    style={{ '--ticker-copies': copies }}
                >
                    {Array.from({ length: copies }, (_, i) => (
                        <div
                            key={i}
                            ref={i === 0 ? measureRef : undefined}
                            className="figma-stroke-ticker__inner"
                        >
                            <TickerRow />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
